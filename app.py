from flask import Flask, request, jsonify, render_template
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
    
        output = model(image)
    
  
    return output

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            image = Image.open(file.stream)
            image = transform(image).unsqueeze(0)
            output = predict(model, image)
            return jsonify(output)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
