import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import matplotlib.pyplot as plt
import os

# -----------------------------
# CONFIG
# -----------------------------
train_dir = r"C:\Users\User\indian_bovine_split\train"
val_dir = r"C:\Users\User\indian_bovine_split\val"
batch_size = 32   # keep even number
num_epochs_stage1 = 5
num_epochs_stage2 = 5
num_epochs_stage3 = 10
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = r"d:\Python basics\AI project\basics.py\best_vit_model.pth"

# -----------------------------
# DATA
# -----------------------------
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_tfms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)

# -----------------------------
# MODEL
# -----------------------------
model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=num_classes)
model = model.to(device)

# Mixup + criterion
mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=num_classes)
criterion_soft = SoftTargetCrossEntropy()
criterion = nn.CrossEntropyLoss()

# -----------------------------
# TRAINING UTILS
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, mixup_fn, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Apply mixup only if batch is even
        if mixup_fn and imgs.size(0) % 2 == 0:
            imgs, labels = mixup_fn(imgs, labels)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        if not isinstance(labels, torch.Tensor) or labels.ndim == 1:
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples


# -----------------------------
# TRAINING STAGES
# -----------------------------
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_acc = 0.0

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# ---- Stage 1: Train head only ----
print("\n[Stage 1] Training head only...")
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

for epoch in range(num_epochs_stage1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion_soft, optimizer, mixup_fn, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch+1}/{num_epochs_stage1}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_save_path)

# ---- Stage 2: Fine-tune last 8 blocks ----
print("\n[Stage 2] Fine-tuning last 8 blocks...")
for param in model.parameters():
    param.requires_grad = False
for blk in model.blocks[-8:]:
    for param in blk.parameters():
        param.requires_grad = True
for param in model.head.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate/5)

for epoch in range(num_epochs_stage2):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion_soft, optimizer, mixup_fn, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch+1}/{num_epochs_stage2}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_save_path)

# ---- Stage 3: Fine-tune all layers ----
print("\n[Stage 3] Fine-tuning all layers...")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=learning_rate/10)

for epoch in range(num_epochs_stage3):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion_soft, optimizer, mixup_fn, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch+1}/{num_epochs_stage3}: train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_save_path)

print(f"\n✅ Training complete! Best val_acc={best_acc:.4f}, model saved to {model_save_path}")

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")

plt.show()
