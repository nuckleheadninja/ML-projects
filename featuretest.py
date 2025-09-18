import torch
import timm
from torchvision import transforms
from PIL import Image

# ------------------------------
# CONFIG
# ------------------------------
NUM_CLASSES = 41
MODEL_PATH = r"D:\Python basics\AI project\basics.py\bovine_breed_classifier.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Define class names
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
# Breed features dictionary
# ------------------------------
breed_features = {
    "Gir": {"avg_weight": "400–475 kg", "avg_height": "130–140 cm", 
            "milk_yield_day": "12–15 L/day", "milk_yield_lactation": "2000–3000 L",
            "lifespan": "12–15 years", "climate": "Hot, dry regions", "best_state": "Gujarat"},
    
    "Murrah": {"avg_weight": "550–650 kg", "avg_height": "132–142 cm", 
               "milk_yield_day": "8–16 L/day", "milk_yield_lactation": "1500–2500 L",
               "lifespan": "15 years", "climate": "All climates, heat-tolerant", "best_state": "Haryana, Punjab"},
    
    "Sahiwal": {"avg_weight": "425–500 kg", "avg_height": "135–140 cm", 
                "milk_yield_day": "8–10 L/day", "milk_yield_lactation": "1800–2500 L",
                "lifespan": "15–20 years", "climate": "Hot, humid", "best_state": "Punjab, Haryana, UP"},
    "Red_Sindhi": {
        "weight_kg": {"female": 325, "male": 530},
        "height_cm": {"female": 115, "male": 132},
        "milk_yield_lpd": 12,
        "lifespan_years": 12,
        "climate": "Heat- and disease-resistant",
        "best_states": ["Gujarat", "Rajasthan"]
    },
    "Tharparkar": {
        "weight_kg": {"male": 430, "female": 310},
        "height_cm": {"male": 143.5, "female": 133.5},
        "milk_yield_lpd": 12.5,
        "lifespan_years": 12,
        "climate": "Arid, desert-adapted",
        "best_states": ["Rajasthan"]
    },
    "Bhadawari": {
        "weight_kg": {"male": 475, "female": 425},
        "height_cm": {"male": 128, "female": 124},
        "milk_yield_lactation_l": 752,
        "lifespan_years": 12,
        "climate": "Riverine semi-humid",
        "best_states": ["Uttar Pradesh", "Madhya Pradesh"]
    },
    "Banni": {
        "milk_yield_lpd": 16,
        "milk_yield_lactation_l": 2600,
        "lifespan_years": 12,
        "climate": "Arid, drought-prone",
        "best_states": ["Gujarat"]
    },
    "Hariana": {
        "milk_yield_lpd": 12.5,
        "lifespan_years": 12,
        "climate": "Semi-arid North India",
        "best_states": ["Haryana"]
    },
    "Kangayam": {
        "weight_kg": {"male": 523, "female": 341},
        "height_cm": {"male": 140, "female": 125},
        "milk_yield_lpd": None,
        "lifespan_years": 12,
        "climate": "Tropical, drought-prone",
        "best_states": ["Tamil Nadu"]
    }
    # Add rest of breeds here...
}

# ------------------------------
# Model
# ------------------------------
model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ------------------------------
# Transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------------
# Predict function
# ------------------------------
def predict_image(image_path, model, transform, class_names):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)

    breed = class_names[preds.item()]
    return breed

# ------------------------------
# Run prediction
# ------------------------------
test_image = r"C:\Users\User\Downloads\imagesCAT.jpg" # Change path to your test image
predicted_breed = predict_image(test_image, model, transform, class_names)

print(f"\n✅ Predicted Breed: {predicted_breed}")

if predicted_breed in breed_features:
    details = breed_features[predicted_breed]
    print("\n📌 Breed Details:")
    for k, v in details.items():
        print(f"  {k}: {v}")
else:
    print("ℹ️ No extra details available for this breed.")
