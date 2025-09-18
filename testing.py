import torch
import timm
from torchvision import transforms, datasets
from PIL import Image

# Paths
DATA_DIR = r"C:\Users\User\.cache\kagglehub\datasets\lukex9442\indian-bovine-breeds\versions\1\Indian_bovine_breeds"
#MODEL_PATH = "D:\Python basics\AI project\basics.py\bovine_breed_classifier.pth"   # Path where you saved your model
MODEL_PATH = r"D:\Python basics\AI project\basics.py\bovine_breed_classifier.pth"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Load dataset only to get class names
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = dataset.classes
NUM_CLASSES = len(class_names)

# ✅ Create model
model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ✅ Prediction function
def predict_image(image_path, model, transform, class_names):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]

# ✅ Test on one image
test_image = r"C:\Users\User\Downloads\Sahiwal-cow2.webp"
predicted_breed = predict_image(test_image, model, transform, class_names)

print("Predicted Breed:", predicted_breed)

