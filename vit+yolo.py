# -------------------------------------------
# YOLO + ViT Cow Breed Classifier
# -------------------------------------------

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import timm
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = r"C:\Users\User\cow_dataset"   # original dataset with folders by breed
CROP_DIR = r"C:\Users\User\cow_dataset_cropped"  # YOLO-cropped dataset
BATCH_SIZE = 32
NUM_CLASSES = 41
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# STEP 1: YOLO Detection & Cropping
# -----------------------------
def crop_with_yolo():
    model = YOLO("yolov8s.pt")  # pretrained general model
    os.makedirs(CROP_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(DATA_DIR, split)
        crop_split_dir = os.path.join(CROP_DIR, split)
        os.makedirs(crop_split_dir, exist_ok=True)

        for breed in os.listdir(split_dir):
            breed_dir = os.path.join(split_dir, breed)
            crop_breed_dir = os.path.join(crop_split_dir, breed)
            os.makedirs(crop_breed_dir, exist_ok=True)

            for img_name in os.listdir(breed_dir):
                img_path = os.path.join(breed_dir, img_name)
                results = model(img_path, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()

                if len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])  # take first cow
                    from PIL import Image
                    img = Image.open(img_path).convert("RGB")
                    cropped = img.crop((x1, y1, x2, y2))
                    cropped.save(os.path.join(crop_breed_dir, img_name))
                else:
                    # if YOLO fails, copy original
                    shutil.copy(img_path, os.path.join(crop_breed_dir, img_name))

# Uncomment this once to crop dataset
# crop_with_yolo()

# -----------------------------
# STEP 2: Data Augmentations
# -----------------------------
train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_ds = datasets.ImageFolder(os.path.join(CROP_DIR, "train"), transform=train_tfm)
val_ds = datasets.ImageFolder(os.path.join(CROP_DIR, "val"), transform=val_tfm)
test_ds = datasets.ImageFolder(os.path.join(CROP_DIR, "test"), transform=val_tfm)

# Handle imbalance with Weighted Sampler
class_counts = torch.bincount(torch.tensor(train_ds.targets))
weights = 1. / class_counts
sample_weights = weights[train_ds.targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -----------------------------
# STEP 3: Vision Transformer
# -----------------------------
model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

# -----------------------------
# Training function
# -----------------------------
def train_stage(model, train_loader, val_loader, epochs, lr, unfreeze_layers=None):
    if unfreeze_layers:
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        correct, total, train_loss = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss /= total

        # Validation
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total

        print(f"[E{epoch+1}/{epochs}] train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "vit_best.pth")

        scheduler.step()

# -----------------------------
# STAGE 1: Train head only
# -----------------------------
print("🔑 Stage 1: Training head only...")
train_stage(model, train_loader, val_loader, epochs=5, lr=1e-3, unfreeze_layers=["head"])

# -----------------------------
# STAGE 2: Fine-tune last 8 blocks
# -----------------------------
print("🔓 Stage 2: Fine-tuning last 8 blocks...")
train_stage(model, train_loader, val_loader, epochs=30, lr=5e-5, unfreeze_layers=[f"blocks.{i}" for i in range(8, 12)] + ["head"])

# -----------------------------
# STAGE 3: Full fine-tuning
# -----------------------------
print("🚀 Stage 3: Full backbone fine-tuning...")
train_stage(model, train_loader, val_loader, epochs=20, lr=1e-5)

# -----------------------------
# Evaluation
# -----------------------------
print("✅ Loading best model and evaluating...")
model.load_state_dict(torch.load("vit_best.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=test_ds.classes))
