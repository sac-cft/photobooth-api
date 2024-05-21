from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
from insightface.app import FaceAnalysis
import insightface
import cv2
import os
from typing import List
import uuid
import random
import threading
import time

app = FastAPI()

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

import gdown

# Download 'inswapper_128.onnx' file using gdown
# model_url = 'https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu'
# model_output_path = 'inswapper/inswapper_128.onnx'
# if not os.path.exists(model_output_path):
#     gdown.download(model_url, model_output_path, quiet=False)

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Directory setup
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
SOURCE_FOLDER = 'source_images'
TARGET_FOLDER = 'target_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(SOURCE_FOLDER, exist_ok=True)
os.makedirs(TARGET_FOLDER, exist_ok=True)

def simple_face_swap(sourceImage, targetImage, face_app, swapper):
    facesimg1 = face_app.get(sourceImage)
    facesimg2 = face_app.get(targetImage)
    
    if len(facesimg1) == 0 or len(facesimg2) == 0:
        return None  # No faces detected
    
    face1 = facesimg1[0]
    face2 = facesimg2[0]

    img1_swapped = swapper.get(sourceImage, face1, face2, paste_back=True)
    
    return img1_swapped

def process_target_image(target_image_path):
    source_files = os.listdir(SOURCE_FOLDER)
    source_filename = random.choice(source_files)
    source_image_path = os.path.join(SOURCE_FOLDER, source_filename)
    target_image = cv2.imread(target_image_path)
    source_image = cv2.imread(source_image_path)

    swapped_image = simple_face_swap(source_image, target_image, face_app, swapper)
    if swapped_image is not None:
        result_filename = os.path.splitext(os.path.basename(target_image_path))[0] + '_swapped.jpg'
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, swapped_image)

def watch_target_folder():
    target_before = dict([(f, None) for f in os.listdir(TARGET_FOLDER)])
    while True:
        target_after = dict([(f, None) for f in os.listdir(TARGET_FOLDER)])
        
        target_added = [f for f in target_after if not f in target_before]
        
        if target_added:
            for filename in target_added:
                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                    target_image_path = os.path.join(TARGET_FOLDER, filename)
                    process_target_image(target_image_path)

        target_before = target_after
        time.sleep(1)

# Watch target folder in a separate thread
watch_thread = threading.Thread(target=watch_target_folder)
watch_thread.daemon = True
watch_thread.start()

@app.post("/api/swap-face/")
async def swap_faces(sourceImage: UploadFile = File(...), targetImage: UploadFile = File(...)):
    img1_path = os.path.join(UPLOAD_FOLDER, sourceImage.filename)
    img2_path = os.path.join(UPLOAD_FOLDER, targetImage.filename)

    with open(img1_path, "wb") as buffer:
        shutil.copyfileobj(sourceImage.file, buffer)
    with open(img2_path, "wb") as buffer:
        shutil.copyfileobj(targetImage.file, buffer)

    sourceImage = cv2.imread(img1_path)
    targetImage = cv2.imread(img2_path)

    swapped_image = simple_face_swap(sourceImage, targetImage, face_app, swapper)
    if swapped_image is None:
        raise HTTPException(status_code=500, detail="Face swap failed")

    result_filename = str(uuid.uuid4()) + '.jpg'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, swapped_image)

    return FileResponse(result_path)

# HTTP
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
