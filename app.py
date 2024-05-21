from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

model_path = 'model_path.pth'  # Replace with your actual model path
model = load_model(model_path)

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define the predict function
def predict(model, image):
    # Make prediction
    with torch.no_grad():
        output = model(image)
    
    # Assuming the model returns bounding boxes and scores
    boxes = output[0]['boxes'].tolist()
    scores = output[0]['scores'].tolist()

    # Filter out boxes with low scores (e.g., confidence threshold of 0.5)
    threshold = 0.5
    result = []
    for box, score in zip(boxes, scores):
        if score >= threshold:
            result.append({'box': box, 'score': score})

    return result

@app.route('/', methods=['GET'])
def home():
    return "Car Damage Detection API"

@app.route('/predict', methods=['POST'])
def predict_damage():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)

    # Transform the image
    image = transform(image).unsqueeze(0)

    # Get prediction
    output = predict(model, image)
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
