from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import subprocess
import insightface
from insightface.app import FaceAnalysis
import gdown
import cv2
import uuid

app = Flask(__name__)

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Download and load the inswapper model
model_url = 'https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu'
model_output_path = 'inswapper/inswapper_128.onnx'
if not os.path.exists(model_output_path):
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    gdown.download(model_url, model_output_path, quiet=False)

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Configure the application
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def simple_face_swap(img1, img2, face_app, swapper):
    facesimg1 = face_app.get(img1)
    facesimg2 = face_app.get(img2)
    
    if len(facesimg1) == 0 or len(facesimg2) == 0:
        return None  # No faces detected
    
    face1 = facesimg1[0]
    face2 = facesimg2[0]

    img1_swapped = swapper.get(img1, face1, face2, paste_back=True)
    
    return img1_swapped

@app.route('/api/swap-face', methods=['POST'])
def swap_faces():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({'error': 'Please provide two images'}), 400

    img1_file = request.files['img1']
    img2_file = request.files['img2']

    if img1_file.filename == '' or img2_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img1_file.filename))
    img2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img2_file.filename))

    img1_file.save(img1_path)
    img2_file.save(img2_path)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    swapped_image = simple_face_swap(img1, img2, face_app, swapper)
    if swapped_image is None:
        return jsonify({'error': 'Face swap failed'}), 500

    result_filename = str(uuid.uuid4()) + '.jpg'
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, swapped_image)

    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
