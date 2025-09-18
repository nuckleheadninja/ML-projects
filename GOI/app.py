import gradio as gr
import torch
import timm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


MODEL_SAVE_PATH = r"D:\Python basics\AI project\vit_small_bovine_best_stage3.pth"

NUM_CLASSES     = 41
IMG_SIZE        = 224
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
   'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 
    'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 
    'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 
    'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 
    'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 
    'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 
    'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur'
]
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


model = load_trained_model(MODEL_SAVE_PATH, NUM_CLASSES)


val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(input_image: Image.Image):
    """
    Takes a PIL image, processes it, and returns the top 3 predictions.
    """
    if input_image is None:
        return None

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

    return top3_preds

if __name__ == "__main__":
    image_input = gr.Image(type="pil", label="Upload Bovine Image")
    output_label = gr.Label(num_top_classes=3, label="Top 3 Predictions")

    iface = gr.Interface(
        fn=predict,
        inputs=image_input,
        outputs=output_label,
        title="🐄 Indian Bovine Breed Classifier",
        description="Upload an image of an Indian bovine (cow, bull, etc.) to classify its breed using a ViT-Small model.",
        examples=[
            [r"C:\Users\User\indian_bovine_split\test\Holstein_Friesian\Holstein_Friesian_106.jpg"
] 
        ]
    )

    

    iface.launch()
