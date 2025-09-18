
import gradio as gr
import torch
import timm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# ------------------------------
# BREED FEATURES DICTIONARY (Provided by you)
# ------------------------------
breed_features = {
    "Gir": { "type": "Cattle", "avg_weight_kg": {"male": 545, "female": 385}, "avg_height_cm": {"male": 135, "female": 130}, "milk_yield_lactation_l": 2110, "lifespan_years": 12, "climate": "Hot and dry regions", "best_state": "Gujarat" },
    "Murrah": { "type": "Buffalo", "avg_weight_kg": {"male": 550, "female": 450}, "avg_height_cm": {"male": 142, "female": 132}, "milk_yield_lactation_l": 1800, "lifespan_years": 12, "climate": "All climates, heat-tolerant", "best_state": "Haryana, Punjab" },
    "Sahiwal": { "type": "Cattle", "avg_weight_kg": {"male": 500, "female": 425}, "avg_height_cm": {"male": 140, "female": 135}, "milk_yield_lactation_l": 2300, "lifespan_years": 15, "climate": "Hot and humid", "best_state": "Punjab, Haryana" },
    "Red_Sindhi": { "type": "Cattle", "avg_weight_kg": {"male": 450, "female": 325}, "avg_height_cm": {"male": 132, "female": 115}, "milk_yield_lactation_l": 1800, "lifespan_years": 12, "climate": "Heat and disease-resistant", "best_state": "Punjab, Haryana" },
    "Tharparkar": { "type": "Cattle", "avg_weight_kg": {"male": 430, "female": 310}, "avg_height_cm": {"male": 143, "female": 133}, "milk_yield_lactation_l": 1750, "lifespan_years": 12, "climate": "Arid, desert-adapted", "best_state": "Rajasthan" },
    "Bhadawari": { "type": "Buffalo", "avg_weight_kg": {"male": 475, "female": 425}, "avg_height_cm": {"male": 128, "female": 124}, "milk_yield_lactation_l": 1200, "lifespan_years": 12, "climate": "Riverine semi-humid", "best_state": "Uttar Pradesh, Madhya Pradesh" },
    "Banni": { "type": "Buffalo", "avg_weight_kg": {"male": 500, "female": 400}, "avg_height_cm": {"male": 130, "female": 125}, "milk_yield_lactation_l": 2600, "lifespan_years": 12, "climate": "Arid, drought-prone", "best_state": "Gujarat" },
    "Hariana": { "type": "Cattle", "avg_weight_kg": {"male": 500, "female": 350}, "avg_height_cm": {"male": 150, "female": 140}, "milk_yield_lactation_l": 1150, "lifespan_years": 12, "climate": "Semi-arid North India", "best_state": "Haryana" },
    "Kangayam": { "type": "Cattle", "avg_weight_kg": {"male": 523, "female": 341}, "avg_height_cm": {"male": 140, "female": 125}, "milk_yield_lactation_l": 550, "lifespan_years": 12, "climate": "Tropical, drought-prone", "best_state": "Tamil Nadu" },
    "Alambadi": { "type": "Cattle", "avg_weight_kg": {"male": 350, "female": 300}, "avg_height_cm": {"male": 125, "female": 115}, "milk_yield_lactation_l": 300, "lifespan_years": 10, "climate": "Hilly and tropical", "best_state": "Tamil Nadu" },
    "Amritmahal": { "type": "Cattle", "avg_weight_kg": {"male": 400, "female": 300}, "avg_height_cm": {"male": 135, "female": 125}, "milk_yield_lactation_l": 570, "lifespan_years": 12, "climate": "Tropical, drought-prone", "best_state": "Karnataka" },
    "Ayrshire": { "type": "Exotic Cattle", "avg_weight_kg": {"male": 800, "female": 550}, "avg_height_cm": {"male": 145, "female": 135}, "milk_yield_lactation_l": 7000, "lifespan_years": 10, "climate": "Temperate, adaptable", "best_state": "Punjab, Haryana (crossbreeding)" },
    "Bargur": { "type": "Cattle", "avg_weight_kg": {"male": 340, "female": 280}, "avg_height_cm": {"male": 125, "female": 118}, "milk_yield_lactation_l": 350, "lifespan_years": 10, "climate": "Hilly, forested areas", "best_state": "Tamil Nadu" },
    "Brown_Swiss": { "type": "Exotic Cattle", "avg_weight_kg": {"male": 900, "female": 650}, "avg_height_cm": {"male": 155, "female": 140}, "milk_yield_lactation_l": 9000, "lifespan_years": 12, "climate": "Cool, temperate", "best_state": "Punjab, Himachal Pradesh (crossbreeding)" },
    "Dangi": { "type": "Cattle", "avg_weight_kg": {"male": 350, "female": 250}, "avg_height_cm": {"male": 125, "female": 115}, "milk_yield_lactation_l": 450, "lifespan_years": 12, "climate": "Heavy rainfall areas", "best_state": "Maharashtra, Gujarat" },
    "Deoni": { "type": "Cattle", "avg_weight_kg": {"male": 550, "female": 400}, "avg_height_cm": {"male": 145, "female": 135}, "milk_yield_lactation_l": 1100, "lifespan_years": 12, "climate": "Semi-arid", "best_state": "Maharashtra, Karnataka" },
    "Guernsey": { "type": "Exotic Cattle", "avg_weight_kg": {"male": 700, "female": 500}, "avg_height_cm": {"male": 140, "female": 130}, "milk_yield_lactation_l": 6500, "lifespan_years": 10, "climate": "Mild, temperate", "best_state": "Nilgiris (historical)" },
    "Hallikar": { "type": "Cattle", "avg_weight_kg": {"male": 360, "female": 250}, "avg_height_cm": {"male": 130, "female": 120}, "milk_yield_lactation_l": 550, "lifespan_years": 12, "climate": "Tropical, arid", "best_state": "Karnataka" },
    "Holstein_Friesian": { "type": "Exotic Cattle", "avg_weight_kg": {"male": 1000, "female": 700}, "avg_height_cm": {"male": 160, "female": 150}, "milk_yield_lactation_l": 10000, "lifespan_years": 6, "climate": "Cool climates, requires good management", "best_state": "Punjab, Haryana (crossbreeding)" },
    "Jaffrabadi": { "type": "Buffalo", "avg_weight_kg": {"male": 600, "female": 500}, "avg_height_cm": {"male": 145, "female": 135}, "milk_yield_lactation_l": 2200, "lifespan_years": 12, "climate": "Coastal, semi-arid", "best_state": "Gujarat" },
    "Jersey": { "type": "Exotic Cattle", "avg_weight_kg": {"male": 700, "female": 450}, "avg_height_cm": {"male": 135, "female": 120}, "milk_yield_lactation_l": 5000, "lifespan_years": 12, "climate": "Adaptable, temperate", "best_state": "All states (crossbreeding)" },
    "Kankrej": { "type": "Cattle", "avg_weight_kg": {"male": 575, "female": 475}, "avg_height_cm": {"male": 155, "female": 140}, "milk_yield_lactation_l": 1800, "lifespan_years": 15, "climate": "Arid and semi-arid", "best_state": "Gujarat, Rajasthan" },
    "Kasargod": { "type": "Cattle", "avg_weight_kg": {"male": 150, "female": 120}, "avg_height_cm": {"male": 100, "female": 90}, "milk_yield_lactation_l": 300, "lifespan_years": 10, "climate": "Hot, humid, coastal", "best_state": "Kerala" },
    "Kenkatha": { "type": "Cattle", "avg_weight_kg": {"male": 320, "female": 230}, "avg_height_cm": {"male": 125, "female": 115}, "milk_yield_lactation_l": 300, "lifespan_years": 10, "climate": "Plains, Bundelkhand region", "best_state": "Uttar Pradesh, Madhya Pradesh" },
    "Kherigarh": { "type": "Cattle", "avg_weight_kg": {"male": 400, "female": 300}, "avg_height_cm": {"male": 140, "female": 130}, "milk_yield_lactation_l": 400, "lifespan_years": 12, "climate": "Terai region", "best_state": "Uttar Pradesh" },
    "Khillari": { "type": "Cattle", "avg_weight_kg": {"male": 450, "female": 380}, "avg_height_cm": {"male": 140, "female": 130}, "milk_yield_lactation_l": 450, "lifespan_years": 12, "climate": "Drought-prone, semi-arid", "best_state": "Maharashtra, Karnataka" },
    "Krishna_Valley": { "type": "Cattle", "avg_weight_kg": {"male": 500, "female": 400}, "avg_height_cm": {"male": 150, "female": 135}, "milk_yield_lactation_l": 900, "lifespan_years": 12, "climate": "Black cotton soil region", "best_state": "Karnataka, Maharashtra" },
    "Malnad_gidda": { "type": "Cattle", "avg_weight_kg": {"male": 120, "female": 100}, "avg_height_cm": {"male": 95, "female": 85}, "milk_yield_lactation_l": 250, "lifespan_years": 10, "climate": "Hilly, heavy rainfall", "best_state": "Karnataka" },
    "Mehsana": { "type": "Buffalo", "avg_weight_kg": {"male": 500, "female": 450}, "avg_height_cm": {"male": 140, "female": 130}, "milk_yield_lactation_l": 1900, "lifespan_years": 12, "climate": "Semi-arid", "best_state": "Gujarat" },
    "Nagpuri": { "type": "Buffalo", "avg_weight_kg": {"male": 525, "female": 425}, "avg_height_cm": {"male": 145, "female": 135}, "milk_yield_lactation_l": 1050, "lifespan_years": 12, "climate": "Semi-arid, Central India", "best_state": "Maharashtra" },
    "Nagori": { "type": "Cattle", "avg_weight_kg": {"male": 450, "female": 350}, "avg_height_cm": {"male": 145, "female": 135}, "milk_yield_lactation_l": 600, "lifespan_years": 12, "climate": "Arid, desert", "best_state": "Rajasthan" },
    "Nili_Ravi": { "type": "Buffalo", "avg_weight_kg": {"male": 600, "female": 500}, "avg_height_cm": {"male": 140, "female": 132}, "milk_yield_lactation_l": 1950, "lifespan_years": 12, "climate": "Plains of Punjab", "best_state": "Punjab" },
    "Nimari": { "type": "Cattle", "avg_weight_kg": {"male": 400, "female": 300}, "avg_height_cm": {"male": 135, "female": 125}, "milk_yield_lactation_l": 700, "lifespan_years": 12, "climate": "Narmada valley", "best_state": "Madhya Pradesh" },
    "Ongole": { "type": "Cattle", "avg_weight_kg": {"male": 550, "female": 450}, "avg_height_cm": {"male": 160, "female": 145}, "milk_yield_lactation_l": 1500, "lifespan_years": 15, "climate": "Tropical, adaptable", "best_state": "Andhra Pradesh" },
    "Pulikulam": { "type": "Cattle", "avg_weight_kg": {"male": 400, "female": 300}, "avg_height_cm": {"male": 130, "female": 120}, "milk_yield_lactation_l": 300, "lifespan_years": 10, "climate": "Dry, hot", "best_state": "Tamil Nadu" },
    "Rathi": { "type": "Cattle", "avg_weight_kg": {"male": 400, "female": 300}, "avg_height_cm": {"male": 130, "female": 120}, "milk_yield_lactation_l": 1500, "lifespan_years": 12, "climate": "Arid, desert", "best_state": "Rajasthan" },
    "Red_Dane": { "type": "Exotic Cattle", "avg_weight_kg": {"male": 950, "female": 650}, "avg_height_cm": {"male": 150, "female": 138}, "milk_yield_lactation_l": 8000, "lifespan_years": 10, "climate": "Temperate", "best_state": "Punjab (crossbreeding)" },
    "Surti": { "type": "Buffalo", "avg_weight_kg": {"male": 450, "female": 400}, "avg_height_cm": {"male": 130, "female": 125}, "milk_yield_lactation_l": 1700, "lifespan_years": 12, "climate": "Coastal, tropical", "best_state": "Gujarat" },
    "Toda": { "type": "Buffalo", "avg_weight_kg": {"male": 400, "female": 350}, "avg_height_cm": {"male": 120, "female": 110}, "milk_yield_lactation_l": 500, "lifespan_years": 12, "climate": "High altitude, Nilgiris", "best_state": "Tamil Nadu" },
    "Umblachery": { "type": "Cattle", "avg_weight_kg": {"male": 360, "female": 300}, "avg_height_cm": {"male": 125, "female": 110}, "milk_yield_lactation_l": 450, "lifespan_years": 10, "climate": "Coastal, Cauvery delta", "best_state": "Tamil Nadu" },
    "Vechur": { "type": "Cattle", "avg_weight_kg": {"male": 150, "female": 120}, "avg_height_cm": {"male": 95, "female": 90}, "milk_yield_lactation_l": 560, "lifespan_years": 10, "climate": "Hot, humid, coastal", "best_state": "Kerala" }
}

# ------------------------------
# CONFIGURATION
# ------------------------------
MODEL_SAVE_PATH = r"D:\Python basics\AI project\vit_small_bovine_best_stage3.pth"
NUM_CLASSES = 41
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 
    'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 
    'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 
    'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 
    'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 
    'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 
    'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur'
]

# ------------------------------
# MODEL LOADING
# ------------------------------
def build_model(num_classes):
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    return model

def load_trained_model(model_path, num_classes):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully and set to evaluation mode.")
    return model

model = load_trained_model(MODEL_SAVE_PATH, NUM_CLASSES)

# ------------------------------
# IMAGE TRANSFORMATION
# ------------------------------
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------------------------------
# PREDICTION & FEATURE FUNCTIONS
# ------------------------------
def predict(input_image: Image.Image):
    if input_image is None:
        return None, None

    image_tensor = val_transform(input_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    top3_probs, top3_indices = torch.topk(probabilities, 3)
    
    top3_preds = {}
    for i in range(top3_probs.size(1)):
        prob = top3_probs[0, i].item()
        class_index = top3_indices[0, i].item()
        class_name = CLASS_NAMES[class_index]
        top3_preds[class_name] = f"{prob:.4f}"
        
    # Return predictions and also hide the features button initially
    return top3_preds, gr.update(visible=True)

def get_features(predictions: dict):
    """
    Takes the prediction dictionary and returns a formatted string of features for the top breed.
    """
    if not predictions:
        return "No prediction available to show features."
        
    # Get the top breed (first key in the prediction dictionary)
    top_breed_name = list(predictions.keys())[0]
    
    # Retrieve the features for that breed
    features = breed_features.get(top_breed_name)
    
    if not features:
        return f"No feature data available for {top_breed_name}."
        
    # Format the features into a nice Markdown string
    md_string = f"### Features for {top_breed_name}\n"
    md_string += "---\n"
    for key, value in features.items():
        # Clean up the key for display (e.g., 'avg_weight_kg' -> 'Avg Weight (kg)')
        display_key = key.replace('_', ' ').replace('kg', '(kg)').replace('cm', '(cm)').replace('l', '(L)').title()
        
        # Handle nested dictionaries for weight/height
        if isinstance(value, dict):
            display_value = ', '.join([f"{k.title()}: {v}" for k,v in value.items()])
        else:
            display_value = str(value)
            
        md_string += f"- **{display_key}:** {display_value}\n"
        
    return md_string

# ------------------------------
# GRADIO INTERFACE
# ------------------------------
if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown("# 🐄 Indian Bovine Breed Classifier")
        gr.Markdown("Upload an image of an Indian bovine to classify its breed using a ViT-Small model.")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Bovine Image")
                
                with gr.Row():
                    predict_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                output_label = gr.Label(num_top_classes=3, label="Top 3 Predictions")
                show_features_btn = gr.Button("Show Features", visible=False) # Initially hidden
                feature_output = gr.Markdown(label="Breed Features")

        gr.Markdown("---")
        gr.Markdown("""
                            ### Developed by:
                                - Kartikey
                                - Devansh
                                - Jahnavi
                                - Deepanshu
                                - Milind
                                - Pawan
        """)
        
        # Define the interactions
        predict_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[output_label, show_features_btn]
        )
        
        show_features_btn.click(
            fn=get_features,
            inputs=output_label,
            outputs=feature_output
        )

    # Launch the GUI
    iface.launch()