import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------
# USER CONFIG
# ------------------------------
TRAIN_DIR = r"C:\Users\User\indian_bovine_split\train"
VAL_DIR   = r"C:\Users\User\indian_bovine_split\val"

BATCH_SIZE    = 16
NUM_WORKERS   = 0
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 30
PATIENCE      = 7
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE    = "vit_small_bovine_8blocks_best.pth"
IMG_SIZE      = 224
NUM_CLASSES   = 41

# ------------------------------
# Data Transforms
# ------------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(0, shear=10, scale=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------------------
# Build ViT Model
# ------------------------------
def build_model(num_classes):
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    return model

# ------------------------------
# Training Functions
# ------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    running_loss, running_corrects = 0.0, 0
    dataset_size = len(dataloader.dataset)
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda') if scaler else torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
    return running_loss / dataset_size, running_corrects / dataset_size

def validate_one_epoch(model, dataloader, criterion, scaler):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    dataset_size = len(dataloader.dataset)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast(device_type='cuda') if scaler else torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
    return running_loss / dataset_size, running_corrects / dataset_size

# ------------------------------
# Plotting Helper
# ------------------------------
def plot_history(history):
    epochs = range(1, len(history["train_loss"])+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve"); plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------
# Main Training Function
# ------------------------------
def run_training(train_dir, val_dir):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_transform)

    # Class weighting
    targets = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE=='cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE=='cuda'))

    model = build_model(NUM_CLASSES).to(DEVICE)

    # Stage 1: freeze first 4 blocks, train last 8 + head
    for name, param in model.named_parameters():
        if any(f"blocks.{i}" in name for i in range(0,4)):
            param.requires_grad = False
        else:
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Layer-wise optimizer: head and last 8 blocks
    head_params = [p for n,p in model.named_parameters() if "head" in n]
    last_blocks_params = [p for n,p in model.named_parameters() if any(f"blocks.{i}" in n for i in range(4,12))]
    optimizer = optim.AdamW([
        {"params": head_params, "lr":1e-3},
        {"params": last_blocks_params, "lr":5e-4}
    ], weight_decay=1e-4)

    scaler = torch.amp.GradScaler() if DEVICE=='cuda' else None
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    print("🔓 Training last 8 blocks + head")
    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience_counter = 0

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE2)

    for epoch in range(EPOCHS_STAGE2):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc     = validate_one_epoch(model, val_loader, criterion, scaler)
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)

        scheduler.step()

        print(f"[E{epoch+1}/{EPOCHS_STAGE2}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, MODEL_SAVE)
            print(f"💾 Saved best model (val_loss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹️ Early stopping triggered")
                break

    model.load_state_dict(best_wts)
    return model, history, train_dataset.classes

# ------------------------------
# ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    trained_model, history, classes = run_training(TRAIN_DIR, VAL_DIR)
    print("✅ Training finished. Best model saved to:", MODEL_SAVE)
    plot_history(history)
