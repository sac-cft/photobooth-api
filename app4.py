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
import pygame
from pygame.locals import *

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
    source_img = cv2.imread(sourceImage)
    target_img = cv2.imread(targetImage)

    faces_source = face_app.get(source_img)
    faces_target = face_app.get(target_img)

    if len(faces_source) == 0 or len(faces_target) == 0:
        return None  # No faces detected

    face_source = faces_source[0]
    face_target = faces_target[0]

    img_swapped = swapper.get(target_img, face_target, face_source, paste_back=True)

    return img_swapped

# Function to display images in full screen
def display_image(image_path):
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    image = pygame.image.load(image_path)
    screen.blit(image, (0, 0))
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
                break

    pygame.quit()

# Function to watch the result folder for new images
def watch_result_folder():
    result_before = set(os.listdir(RESULT_FOLDER))
    while True:
        result_after = set(os.listdir(RESULT_FOLDER))

        new_images = result_after - result_before
        for image in new_images:
            image_path = os.path.join(RESULT_FOLDER, image)
            display_image(image_path)

        result_before = result_after

        # Check for new images every second
        time.sleep(1)

# Watch result folder in a separate thread
watch_thread = threading.Thread(target=watch_result_folder)
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

    swapped_image = simple_face_swap(img1_path, img2_path, face_app, swapper)
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
