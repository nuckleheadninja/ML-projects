import gradio as gr
import torch
import timm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# ------------------------------
# CONFIGURATION
# ------------------------------
# --- MUST BE THE SAME AS IN YOUR TRAINING SCRIPT ---
MODEL_SAVE_PATH = r"D:\Python basics\AI project\vit_small_bovine_best_stage3.pth"

NUM_CLASSES     = 41
IMG_SIZE        = 224
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# --- IMPORTANT: You must provide your class names here ---
# You can get this list from your training script by printing train_dataset.classes
# The order MUST match the order used during training.
CLASS_NAMES = [
   'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 
    'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 
    'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 
    'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 
    'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 
    'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 
    'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur'
] # <-- REPLACE WITH YOUR ACTUAL CLASS NAMES IN ORDER

# ------------------------------
# MODEL LOADING
# ------------------------------

def build_model(num_classes):
    """Builds the ViT model architecture."""
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
    return model

def load_trained_model(model_path, num_classes):
    """Loads the model architecture and the saved weights."""
    model = build_model(num_classes)
    # Load the state dictionary. Use map_location for CPU if CUDA isn't available.
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval() # Set the model to evaluation mode
    print("✅ Model loaded successfully and set to evaluation mode.")
    return model

# Load the model when the script starts
model = load_trained_model(MODEL_SAVE_PATH, NUM_CLASSES)

# ------------------------------
# IMAGE TRANSFORMATION
# ------------------------------

# Use the same validation transform as in your training script
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------

def predict(input_image: Image.Image):
    """
    Takes a PIL image, processes it, and returns the top 3 predictions.
    """
    if input_image is None:
        return None

    # Apply transformations
    image_tensor = val_transform(input_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Get raw model outputs (logits)
        outputs = model(image_tensor)
        # Convert logits to probabilities using softmax
        probabilities = F.softmax(outputs, dim=1)

    # Get the top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, 3)

    # Format the output as a dictionary for Gradio's Label component
    top3_preds = {}
    for i in range(top3_probs.size(1)):
        prob = top3_probs[0, i].item()
        class_index = top3_indices[0, i].item()
        class_name = CLASS_NAMES[class_index]
        top3_preds[class_name] = f"{prob:.4f}" # Format as a string with 4 decimal places

    return top3_preds

# ------------------------------
# GRADIO INTERFACE
# ------------------------------
if __name__ == "__main__":
    # Define the input and output components for the GUI
    image_input = gr.Image(type="pil", label="Upload Bovine Image")
    output_label = gr.Label(num_top_classes=3, label="Top 3 Predictions")

    # Create the Gradio interface
    iface = gr.Interface(
        fn=predict,
        inputs=image_input,
        outputs=output_label,
        title="🐄 Indian Bovine Breed Classifier",
        description="Upload an image of an Indian bovine (cow, bull, etc.) to classify its breed using a ViT-Small model.",
        examples=[
            [r"C:\Users\User\indian_bovine_split\test\Holstein_Friesian\Holstein_Friesian_106.jpg"
] # Add paths to some example images if you have them
        ]
    )

    # Launch the GUI
    iface.launch()