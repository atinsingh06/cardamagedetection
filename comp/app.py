# app.py
from flask import Flask, request, render_template, send_from_directory
import os
from model import generate_bbox
import cv2
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

@app.route('/dent-detect', methods=[ 'POST', 'GET'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        output_image = generate_bbox(file_path)
        output_image_path = 'static/output_image.jpg'
        cv2.imwrite(output_image_path, output_image)  # Save the processed image

        return {'output': output_image_path}



if __name__ == '__main__':
    app.run(debug=True)
