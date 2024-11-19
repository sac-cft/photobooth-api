# from fastapi import FastAPI, File, UploadFile, HTTPException, Form
# from fastapi.responses import FileResponse, JSONResponse
# import shutil
# from insightface.app import FaceAnalysis
# import insightface
# import cv2
# import os
# import uuid
# from fastapi import FastAPI, File, UploadFile

# app = FastAPI()
# # Initialize FaceAnalysis
# face_app = FaceAnalysis(name='buffalo_l')
# face_app.prepare(ctx_id=0, det_size=(640, 640))

# swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# # Directory setup
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# UPLOAD_CHAR = os.path.abspath(r"C:\Users\sachi\Desktop\AI Swap Standardization\saas-ai-photobooth-front-end\public")
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
# os.makedirs(UPLOAD_CHAR, exist_ok=True)

# def simple_face_swap(sourceImage, targetImage, face_app, swapper):
#     facesimg1 = face_app.get(sourceImage)
#     facesimg2 = face_app.get(targetImage)
    
#     if len(facesimg1) == 0 or len(facesimg2) == 0:
#         return None  # No faces detected
    
#     face1 = facesimg1[0]
#     face2 = facesimg2[0]

#     img1_swapped = swapper.get(sourceImage, face1, face2, paste_back=True)
    
#     return img1_swapped

# @app.post("/api/swap-face/")
# async def swap_faces(sourceImage: UploadFile = File(...), targetImage: UploadFile = File(...)):
#     img1_path = os.path.join(UPLOAD_FOLDER, sourceImage.filename)
#     img2_path = os.path.join(UPLOAD_FOLDER, targetImage.filename)

#     with open(img1_path, "wb") as buffer:
#         shutil.copyfileobj(sourceImage.file, buffer)
#     with open(img2_path, "wb") as buffer:
#         shutil.copyfileobj(targetImage.file, buffer)

#     sourceImage = cv2.imread(img1_path)
#     targetImage = cv2.imread(img2_path)

#     swapped_image = simple_face_swap(sourceImage, targetImage, face_app, swapper)
#     if swapped_image is None:
#         raise HTTPException(status_code=500, detail="Face swap failed")

#     result_filename = str(uuid.uuid4()) + '.jpg'
#     result_path = os.path.join(RESULT_FOLDER, result_filename)
#     cv2.imwrite(result_path, swapped_image)

#     return FileResponse(result_path)

# @app.post("/upload")
# async def upload_image(file: UploadFile = File(...), filename: str = Form(...)):
#     # Replace spaces with underscores
#     filename = filename.replace(" ", "_")
    
#     # Get file extension
#     extension = os.path.splitext(file.filename)[1]
    
#     # Ensure filename has the correct extension
#     if not filename.endswith(extension):
#         filename += extension
    
#     file_location = os.path.join(UPLOAD_CHAR, filename)
#     with open(file_location, "wb") as f:
#         f.write(file.file.read())
    
#     return JSONResponse(content={"filename": filename, "location": file_location})

# # HTTP
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='localhost', port=8000)


# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import FileResponse
# import shutil
# from insightface.app import FaceAnalysis
# import insightface
# import cv2
# import os
# import uuid
# from gfpgan import GFPGANer
# import numpy as np
# from PIL import Image
# import logging

# app = FastAPI()

# # Initialize logging
# logging.basicConfig(level=logging.INFO)

# # Initialize FaceAnalysis
# face_app = FaceAnalysis(name='buffalo_l')
# face_app.prepare(ctx_id=0, det_size=(640, 640))

# swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# # Initialize GFPGAN for face enhancement
# gfpganer = GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2)

# # Directory setup
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# def simple_face_swap(sourceImage, targetImage, face_app, swapper):
#     logging.info("Starting face swap...")
#     facesimg1 = face_app.get(sourceImage)
#     facesimg2 = face_app.get(targetImage)
    
#     logging.info(f"Number of faces detected in source image: {len(facesimg1)}")
#     logging.info(f"Number of faces detected in target image: {len(facesimg2)}")

#     if len(facesimg1) == 0 or len(facesimg2) == 0:
#         return None  # No faces detected
    
#     face1 = facesimg1[0]
#     face2 = facesimg2[0]

#     img1_swapped = swapper.get(sourceImage, face1, face2, paste_back=True)
    
#     logging.info("Face swap completed.")
#     return img1_swapped

# def enhance_face(image):
#     logging.info("Starting face enhancement...")
#     _, _, restored_img = gfpganer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    
#     logging.info(f"Type of restored_img: {type(restored_img)}")
#     if isinstance(restored_img, Image.Image):
#         restored_img = np.array(restored_img)
#     logging.info(f"Type after conversion (if any): {type(restored_img)}")
#     if isinstance(restored_img, np.ndarray):
#         logging.info("Face enhancement completed.")
#         return restored_img
#     else:
#         raise ValueError("Enhanced image is not a valid numpy array")

# @app.post("/api/swap-face/")
# async def swap_faces(sourceImage: UploadFile = File(...), targetImage: UploadFile = File(...)):
#     img1_path = os.path.join(UPLOAD_FOLDER, sourceImage.filename)
#     img2_path = os.path.join(UPLOAD_FOLDER, targetImage.filename)

#     with open(img1_path, "wb") as buffer:
#         shutil.copyfileobj(sourceImage.file, buffer)
#     with open(img2_path, "wb") as buffer:
#         shutil.copyfileobj(targetImage.file, buffer)

#     sourceImage_cv = cv2.imread(img1_path)
#     targetImage_cv = cv2.imread(img2_path)

#     if sourceImage_cv is None:
#         raise HTTPException(status_code=500, detail=f"Failed to read source image with OpenCV: {img1_path}")
#     if targetImage_cv is None:
#         raise HTTPException(status_code=500, detail=f"Failed to read target image with OpenCV: {img2_path}")

#     logging.info(f"Source image shape: {sourceImage_cv.shape}")
#     logging.info(f"Target image shape: {targetImage_cv.shape}")

#     swapped_image = simple_face_swap(sourceImage_cv, targetImage_cv, face_app, swapper)
#     if swapped_image is None:
#         raise HTTPException(status_code=500, detail="Face swap failed")

#     logging.info(f"Swapped image shape: {swapped_image.shape}")

#     enhanced_image = enhance_face(swapped_image)

#     logging.info(f"Enhanced image shape: {enhanced_image.shape}")

#     result_filename = str(uuid.uuid4()) + '.jpg'
#     result_path = os.path.join(RESULT_FOLDER, result_filename)
#     cv2.imwrite(result_path, enhanced_image)

#     logging.info(f"Image saved to: {result_path}")

#     return FileResponse(result_path)

# # HTTP server
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='localhost', port=8000)
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
from insightface.app import FaceAnalysis
import insightface
import cv2
import os
import uuid
from gfpgan import GFPGANer
import numpy as np
from PIL import Image
import logging
import requests

app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Initialize GFPGAN for face enhancement
gfpganer = GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2)

# Directory setup
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Function to fetch and save the target image using Clipdrop
def generate_target_image(custom_prompts=None):
    clipdrop_api_key = '2cac03e37041e25b2d2931b8e4f5dc991d946f651f0062251f6007c50060f482953b8b19cd3fdee27c85b0d2b51bedb9'  # Replace with your actual Clipdrop API key
    predefined_prompts_str = "photorealistic concept art, high quality digital art, cinematic, hyperrealism, photorealism, Nikon D850, 8K., sharp focus, emitting diodes, artillery, motherboard, by pascal blanche rutkowski repin artstation hyperrealism painting concept art of detailed character design matte painting, 4 k resolution"

    all_prompts = predefined_prompts_str
    if custom_prompts:
        all_prompts += "\n" + custom_prompts

    clipdrop_url = 'https://clipdrop-api.co/text-to-image/v1'
    headers = {
        'x-api-key': clipdrop_api_key,
        'accept': 'image/webp',
        'x-clipdrop-width': '400',  # Desired width in pixels
        'x-clipdrop-height': '600',  # Desired height in pixels
    }

    data = {
        'prompt': (None, all_prompts, 'text/plain')
    }

    response = requests.post(clipdrop_url, files=data, headers=headers)
    if response.ok:
        target_image_path = os.path.join(UPLOAD_FOLDER, 'target_image.webp')
        with open(target_image_path, 'wb') as f:
            f.write(response.content)
        logging.info(f"Target image generated and saved to {target_image_path}")
        return target_image_path
    else:
        logging.error("Failed to fetch target image from Clipdrop")
        raise HTTPException(status_code=500, detail="Failed to fetch target image from Clipdrop")

def simple_face_swap(sourceImage, targetImage, face_app, swapper):
    logging.info("Starting face swap...")
    sourceImageFace = face_app.get(sourceImage)
    targetImageFace = face_app.get(targetImage)
    
    logging.info(f"Number of faces detected in source image: {len(sourceImageFace)}")
    logging.info(f"Number of faces detected in target image: {len(targetImageFace)}")

    if len(sourceImageFace) == 0 or len(targetImageFace) == 0:
        return None  # No faces detected
    
    sourceImageFaceSelected = sourceImageFace[0]
    targetImageFaceSelected = targetImageFace[0]

    img1_swapped = swapper.get(targetImage, targetImageFaceSelected,sourceImageFaceSelected, paste_back=True)
    
    logging.info("Face swap completed.")
    return img1_swapped

def enhance_face(image):
    logging.info("Starting face enhancement...")
    _, _, restored_img = gfpganer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    
    logging.info(f"Type of restored_img: {type(restored_img)}")
    if isinstance(restored_img, Image.Image):
        restored_img = np.array(restored_img)
    logging.info(f"Type after conversion (if any): {type(restored_img)}")
    if isinstance(restored_img, np.ndarray):
        logging.info("Face enhancement completed.")
        return restored_img
    else:
        raise ValueError("Enhanced image is not a valid numpy array")

@app.post("/api/swap-face/")
async def swap_faces(sourceImage: UploadFile = File(...), prompt: str = Form("")):
    print(prompt)
    print(f"Received prompt: {prompt}")
    img1_path = os.path.join(UPLOAD_FOLDER, sourceImage.filename)

    with open(img1_path, "wb") as buffer:
        shutil.copyfileobj(sourceImage.file, buffer)

    sourceImage_cv = cv2.imread(img1_path)

    if sourceImage_cv is None:
        raise HTTPException(status_code=500, detail=f"Failed to read source image with OpenCV: {img1_path}")

    logging.info(f"Source image shape: {sourceImage_cv.shape}")

    # Fetch target image from Clipdrop
    target_img_path = generate_target_image(prompt)
    targetImage_cv = cv2.imread(target_img_path)

    if targetImage_cv is None:
        raise HTTPException(status_code=500, detail=f"Failed to read target image from Clipdrop: {target_img_path}")

    logging.info(f"Target image shape: {targetImage_cv.shape}")

    swapped_image = simple_face_swap(sourceImage_cv, targetImage_cv, face_app, swapper)
    if swapped_image is None:
        raise HTTPException(status_code=500, detail="Face swap failed")

    logging.info(f"Swapped image shape: {swapped_image.shape}")

    enhanced_image = enhance_face(swapped_image)

    logging.info(f"Enhanced image shape: {enhanced_image.shape}")

    result_filename = str(uuid.uuid4()) + '.jpg'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, enhanced_image)

    logging.info(f"Image saved to: {result_path}")

    return FileResponse(result_path)

# HTTP server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)