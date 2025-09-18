import torch
import torch.nn.functional as F
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------------
# CONFIG
# ------------------------------
NUM_CLASSES = 41
MODEL_PATH = r"D:\Python basics\AI project\basics.py\bovine_breed_classifier.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
IMG_SIZE = 224

# ------------------------------
# Class names (same as training)
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
# Load model
# ------------------------------
model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ------------------------------
# Transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ------------------------------
# Load Test Dataset
# ------------------------------
testdataset_path = r"C:\Users\User\indian_bovine_split\test"  # <- your test set (~100 images)
test_dataset = datasets.ImageFolder(root=testdataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------
# Evaluate on Test Data
# ------------------------------
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ------------------------------
# Precision, Recall, F1
# ------------------------------
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None, labels=range(len(class_names))
)

print("\n📊 Classification Report:")
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
plt.title("Precision, Recall, and F1-score per Class (Test Set)")
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
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.show()
