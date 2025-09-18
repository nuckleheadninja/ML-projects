import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# ------------------------------
# CONFIG
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 15
EPOCHS_STAGE3 = 10
LR = 3e-4
NUM_CLASSES = 41
RAW_DATA_DIR = r"C:\Users\User\indian_bovine_split"

CROP_DIR = r"C:\Users\User\cow_dataset_cropped"
MODEL_SAVE_PATH = "vit_yolo_best.pth"

# ------------------------------
# STEP 1: Run YOLO to crop cows
# ------------------------------
def crop_with_yolo(raw_dir, crop_dir):
    os.makedirs(crop_dir, exist_ok=True)
    yolo = YOLO("yolov8n.pt")  # small YOLO model for detection

    for split in ["train", "val"]:
        split_dir = os.path.join(raw_dir, split)
        out_dir = os.path.join(crop_dir, split)
        os.makedirs(out_dir, exist_ok=True)

        for cls in os.listdir(split_dir):
            cls_in = os.path.join(split_dir, cls)
            cls_out = os.path.join(out_dir, cls)
            os.makedirs(cls_out, exist_ok=True)

            for img_name in os.listdir(cls_in):
                img_path = os.path.join(cls_in, img_name)
                try:
                    results = yolo(img_path)
                    for r in results:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        if len(boxes) > 0:
                            x1, y1, x2, y2 = map(int, boxes[0])
                            img = Image.open(img_path).convert("RGB")
                            crop = img.crop((x1, y1, x2, y2))
                            crop.save(os.path.join(cls_out, img_name))
                except Exception as e:
                    print("Error:", e, img_path)

print("🔎 Running YOLO cropping...")
crop_with_yolo(RAW_DATA_DIR, CROP_DIR)

# ------------------------------
# STEP 2: Dataset & DataLoader
# ------------------------------
train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
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
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------
# STEP 3: Define ViT
# ------------------------------
model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# ------------------------------
# STEP 4: Training loop (3 stages)
# ------------------------------
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
best_val_acc = 0.0

def run_stage(epochs, lr, unfrozen_layers=None):
    global best_val_acc
    if unfrozen_layers is not None:
        for name, param in model.named_parameters():
            param.requires_grad = any(layer in name for layer in unfrozen_layers)
    else:
        for param in model.parameters():
            param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for e in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = validate(model, val_loader)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[E{e+1}/{epochs}] Train Acc={train_acc:.4f} Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("💾 Saved best model!")

print("🔓 Stage 1: Train head only")
for param in model.parameters(): param.requires_grad = False
for param in model.head.parameters(): param.requires_grad = True
run_stage(EPOCHS_STAGE1, LR)

print("🔓 Stage 2: Fine-tune last 8 blocks")
unfrozen = [f"blocks.{i}" for i in range(8, 12)] + ["head"]
run_stage(EPOCHS_STAGE2, LR/3, unfrozen_layers=unfrozen)

print("🔓 Stage 3: Full fine-tuning")
run_stage(EPOCHS_STAGE3, LR/10)

print(f"✅ Training finished. Best model saved at {MODEL_SAVE_PATH}")

# ------------------------------
# STEP 5: Plot graphs
# ------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.legend(); plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.legend(); plt.title("Loss")
plt.show()
