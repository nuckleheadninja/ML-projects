import numpy as np
import pandas as pd
import os
import kagglehub

# Download latest version
path = kagglehub.dataset_download("lukex9442/indian-bovine-breeds")
print("Path to dataset files:", path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# Dataset directory (already downloaded)
DATA_DIR = r"C:\Users\User\.cache\kagglehub\datasets\lukex9442\indian-bovine-breeds\versions\1\Indian_bovine_breeds"

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 15

print("Using device:", DEVICE)

# ✅ Define transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# ✅ Load dataset (just to check classes)
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
NUM_CLASSES = len(full_dataset.classes)

print("Classes:", full_dataset.classes)
print("Number of classes:", NUM_CLASSES)

# ✅ Paths for split datasets
traindataset_path = r"C:\Users\User\indian_bovine_split\train"
validdataset_path = r"C:\Users\User\indian_bovine_split\val"

# ✅ Use ImageFolder for train/val
train_dataset = datasets.ImageFolder(root=traindataset_path, transform=transform)
val_dataset = datasets.ImageFolder(root=validdataset_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ Define model
model = timm.create_model("resnet50", pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ Training functions
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# ✅ Training loop
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
import matplotlib.pyplot as plt

# ✅ Store history
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# ✅ Training loop with history
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# ✅ Plot curves
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

# Plot Train Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history["train_acc"], label="Train Accuracy", marker="o")
plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history["val_loss"], label="Validation Loss", marker="o", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Loss Curve")
plt.legend()

plt.tight_layout()
plt.show()

