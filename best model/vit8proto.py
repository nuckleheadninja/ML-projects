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
from sklearn.metrics import classification_report
import random

TRAIN_DIR = r"C:\Users\User\indian_bovine_split\train"
VAL_DIR   = r"C:\Users\User\indian_bovine_split\val"

BATCH_SIZE    = 16
NUM_WORKERS   = 0
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 30
EPOCHS_STAGE3 = 20
PATIENCE      = 7
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE    = "vit_small_bovine_best_stage3.pth"
IMG_SIZE      = 224
NUM_CLASSES   = 41
MIXUP_ALPHA   = 0.4  #used for stage 2&3 tuning 
#data transforming
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

#building the vit 
def build_model(num_classes):
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    return model

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float).to(DEVICE)

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean() if self.reduction=='mean' else focal_loss.sum()


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#trainig validation starts

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, use_mixup=False):
    model.train()
    running_loss, running_corrects = 0.0, 0
    dataset_size = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
        else:
            targets_a, targets_b, lam = labels, labels, 1.0

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
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
#graph plots
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
#main trainig
def run_training(train_dir, val_dir):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir, transform=val_transform)

    targets = [label for _, label in train_dataset.samples]
    classes = np.unique(targets)
    class_weights = compute_class_weight('balanced', classes=classes, y=targets)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    sample_weights = [class_weights_tensor[label].item() for label in targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE=='cuda'))
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE=='cuda'))

    model = build_model(NUM_CLASSES).to(DEVICE)

   #stage 1 tuning training the head only
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    criterion = FocalLoss(alpha=class_weights_tensor)
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler() if DEVICE=='cuda' else None

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    print("stage 1: training classifier head only (no Mixup)")
    for epoch in range(EPOCHS_STAGE1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, use_mixup=False)
        val_loss, val_acc     = validate_one_epoch(model, val_loader, criterion, scaler)
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss);     history["val_acc"].append(val_acc)
        print(f"[S1 E{epoch+1}/{EPOCHS_STAGE1}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")

   
    # Stage2 head+8 last blocks train
    for name, param in model.named_parameters():
        if any(f"blocks.{i}" in name for i in range(4,12)) or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience_counter = 0

    print("Stage 2: fine-tuning last 8 blocks + head (with Mixup)")
    for epoch in range(EPOCHS_STAGE2):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, use_mixup=True)
        val_loss, val_acc     = validate_one_epoch(model, val_loader, criterion, scaler)
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss);     history["val_acc"].append(val_acc)
        scheduler.step(val_loss)

        print(f"[S2 E{epoch+1}/{EPOCHS_STAGE2}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, MODEL_SAVE)
            print(f" Saved best model (val_loss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping Stage 2")
                break

    model.load_state_dict(best_wts)

   #stage 3 full model f vit small training
    for p in model.parameters():
        p.requires_grad = True  

    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience_counter = 0

    print("Stage 3: full backbone fine-tuning (low LR, Mixup)")
    for epoch in range(EPOCHS_STAGE3):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, use_mixup=True)
        val_loss, val_acc     = validate_one_epoch(model, val_loader, criterion, scaler)
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss);     history["val_acc"].append(val_acc)
        scheduler.step(val_loss)

        print(f"[S3 E{epoch+1}/{EPOCHS_STAGE3}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, MODEL_SAVE)
            print(f"Saved best model Stage 3 (val_loss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping Stage 3")
                break

    model.load_state_dict(best_wts)
    return model, history, train_dataset.classes

# ------------------------------
# ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    trained_model, history, classes = run_training(TRAIN_DIR, VAL_DIR)
    print("Training finished. Best model saved to:", MODEL_SAVE)
    plot_history(history)

        
