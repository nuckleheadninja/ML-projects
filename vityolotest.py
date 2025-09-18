import torch
import timm
from torchvision import transforms
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os

# ------------------------------
# CONFIG
# ------------------------------
NUM_CLASSES = 41
MODEL_PATH = r"vit_small_bovine_best_stage3.pth"   # Your best YOLO+ViT checkpoint
YOLO_WEIGHTS = "yolov8n.pt"  # or your trained YOLO cow detector
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 1   # because YOLO detections are per image

# ------------------------------
# Class names
# ------------------------------
class_names = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari',
    'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana',
    'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej',
    'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley',
    'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi',
    'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi',
    'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur'
]

# ------------------------------
# Load Models
# ------------------------------
yolo = YOLO(YOLO_WEIGHTS)
vit = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
vit.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
vit.to(DEVICE).eval()

# ------------------------------
# Transforms
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ------------------------------
# Test Dataset (folder structure same as ImageFolder)
# ------------------------------
testdataset_path = r"C:\Users\User\indian_bovine_split\test"
classes = sorted(os.listdir(testdataset_path))

all_preds, all_labels = [], []

# ------------------------------
# Run YOLO + ViT evaluation
# ------------------------------
for class_idx, cls in enumerate(classes):
    folder = os.path.join(testdataset_path, cls)
    for fname in os.listdir(folder):
        img_path = os.path.join(folder, fname)
        img = Image.open(img_path).convert("RGB")

        # YOLO detection
        results = yolo(img_path)
        dets = results[0].boxes.xyxy.cpu().numpy()

        if len(dets) == 0:
            # If YOLO fails → skip (or mark as wrong)
            continue

        probs = []
        for (x1, y1, x2, y2) in dets:
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            t = transform(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = vit(t)
                p = torch.softmax(out, dim=1).cpu().numpy()[0]
            probs.append(p)

        # aggregate (probability average)
        avg_prob = np.mean(probs, axis=0)
        pred_idx = int(np.argmax(avg_prob))

        all_preds.append(pred_idx)
        all_labels.append(class_idx)

# ------------------------------
# Metrics
# ------------------------------
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None, labels=range(len(class_names))
)

print("\n📊 Classification Report (YOLO + ViT):")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ------------------------------
# Plot Precision, Recall, F1 per Class
# ------------------------------
x = np.arange(len(class_names))
plt.figure(figsize=(15, 6))
plt.plot(x, precision, marker="o", label="Precision")
plt.plot(x, recall, marker="s", label="Recall")
plt.plot(x, f1, marker="^", label="F1-score")
plt.xticks(x, class_names, rotation=90)
plt.xlabel("Breed")
plt.ylabel("Score")
plt.title("Precision, Recall, F1 per Class - Test Set (YOLO+ViT)")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# Confusion Matrix
# ------------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Test Set (YOLO+ViT)")
plt.tight_layout()
plt.show()
