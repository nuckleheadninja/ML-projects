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
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ------------------------------
# USER CONFIG
# ------------------------------
TRAIN_DIR = r"C:\Users\User\indian_bovine_split\train"
VAL_DIR   = r"C:\Users\User\indian_bovine_split\val"

BATCH_SIZE  = 8        # small batch to fit GPU
NUM_WORKERS = 0
EPOCHS      = 30
PATIENCE    = 5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE  = "vit_small_bovine_best_12blocks.pth"
IMG_SIZE    = 224      # must match pretrained ViT
NUM_CLASSES = 41

# ------------------------------
# Data Transforms
# ------------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
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
# Training/Validation Functions
# ------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    running_loss, running_corrects = 0.0, 0
    dataset_size = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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

            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
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

    # All layers fine-tuned
    for p in model.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = torch.amp.GradScaler() if DEVICE=='cuda' else None

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience_counter = 0

    print("🔓 Stage: fine-tuning all 12 blocks + head")
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc     = validate_one_epoch(model, val_loader, criterion, scaler)

        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss);     history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        print(f"[E{epoch+1}/{EPOCHS}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")

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
