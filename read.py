



# import os
# from flask import Flask, render_template
# from flask_socketio import SocketIO

# app = Flask(__name__)
# socketio = SocketIO(app)

# @app.route('/')
# def index():
#     return 'Server is running'

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')


# @socketio.on('img_send')
# def handle_img_send(data):
#     try:
#         image_data = data.get('imgURL', None)
#         print(data)
#         if image_data is None:
#             raise ValueError('Image data not found in payload')

#         image_path = f'target_images/{os.path.basename(data["imgURL"])}'

#         # Save the image data to a file
#         with open(image_path, 'wb') as f:
#             f.write(image_data.encode())

#         print('Image saved:', image_path)
#     except Exception as e:
#         print('Error saving image:', str(e))

        
# if __name__ == '__main__':
#     socketio.run(app, host='127.0.0.1', port=3001)






# from flask import Flask
# from flask_socketio import SocketIO
# import os
# import base64

# app = Flask(__name__)
# socketio = SocketIO(app)

# @app.route('/')
# def index():
#     return 'Server is running'

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# @socketio.on('img_send')
# def handle_img_send(data):
#     try:
#         base64_image = data.get('imageBase64')
#         print(data)
#         if not base64_image:
#             raise ValueError('Base64 image not found in payload')

#         # Decode base64 string into binary data
#         image_data = base64.b64decode(base64_image)

#         image_path = 'target_images/image.jpg'
#         # Save the image data to a file
#         with open(image_path, 'wb') as f:
#             f.write(image_data)

#         print('Image saved:', image_path)
        
#     except Exception as e:
#         print('Error saving image:', str(e))

# if __name__ == '__main__':
#     socketio.run(app, host='192.168.1.226', port=3001)

























# from flask import Flask
# from flask_socketio import SocketIO
# import os
# import urllib.request
# import base64
# import time

# app = Flask(__name__)
# socketio = SocketIO(app)

# @app.route('/')
# def index():
#     return 'Server is running'

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# @socketio.on('img_send')
# def handle_img_send(data):
#     try:
#         image_base64 = data.get('imageBase64')
#         print(data)
#         if not image_base64:
#             raise ValueError('Image base64 data not found in payload')

#         # Decode base64 data
#         image_data = base64.b64decode(image_base64)

#         # Generate a unique filename based on current timestamp
#         image_name = f"{int(time.time() * 1000)}.jpg"
#         image_path = os.path.join('target_images', image_name)

#         # Save the image to the file
#         with open(image_path, 'wb') as f:
#             f.write(image_data)

#         print('Image saved:', image_path)
#     except Exception as e:
#         print('Error saving image:', str(e))

# if __name__ == '__main__':
#     socketio.run(app, host='127.0.0.1', port=3001)



































from flask import Flask
from flask_socketio import SocketIO
import os
import urllib.request
import base64
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Create directories to save images for different genders
os.makedirs('target_images', exist_ok=True)
os.makedirs('target_images2', exist_ok=True)

@app.route('/')
def index():
    return 'Server is running'

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('img_send')
def handle_img_send(data):
    try:
        image_base64 = data.get('imageBase64')
        gender = data.get('gender')
        print(image_base64)
        
        if not image_base64:
            raise ValueError('Image base64 data not found in payload')

        # Decode base64 data
        image_data = base64.b64decode(image_base64)

        # Generate a unique filename based on current timestamp
        image_name = f"{int(time.time() * 1000)}.jpg"

        # Determine the target directory based on gender
        if gender == 'male':
            target_dir = 'target_images'
        elif gender == 'female':
            target_dir = 'target_images2'
        else:
            raise ValueError('Invalid gender provided')

        image_path = os.path.join(target_dir, image_name)

        # Save the image to the file
        with open(image_path, 'wb') as f:
            f.write(image_data)

        print('Image saved:', image_path)
    except Exception as e:
        print('Error saving image:', str(e))

if __name__ == '__main__':
    socketio.run(app, host='192.168.1.68', port=3001)




















# from flask import Flask
# from flask_socketio import SocketIO
# import os
# import urllib.request
# import base64
# import time

# app = Flask(__name__)
# socketio = SocketIO(app)

# @app.route('/')
# def index():
#     return 'Server is running'

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# import requests
# import base64

# @socketio.on('img_send')
# def handle_img_send(data):
#     try:
#         image_url = data.get('imageBase64')
#         print(data)
#         if not image_url:
#             raise ValueError('Image URL not found in payload')

#         # Fetch image content
#         response = requests.get(image_url)
#         response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes

#         # Encode image content to base64
#         image_base64 = base64.b64encode(response.content).decode('utf-8')

#         # Generate a unique filename based on current timestamp
#         image_name = f"{int(time.time() * 1000)}.jpg"
#         image_path = os.path.join('target_images', image_name)

#         # Save the image to the file
#         with open(image_path, 'wb') as f:
#             f.write(response.content)

#         print('Image saved:', image_path)
#     except Exception as e:
#         print('Error saving image:', str(e))

# if __name__ == '__main__':
#     socketio.run(app, host='192.168.1.68', port=3001)












# from flask import Flask
# from flask_socketio import SocketIO
# import os
# import requests
# import base64
# import time

# app = Flask(__name__)
# socketio = SocketIO(app)

# # Create directories to save images for different genders
# os.makedirs('target_images', exist_ok=True)
# os.makedirs('target_images2', exist_ok=True)

# @app.route('/')
# def index():
#     return 'Server is running'

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')

# @socketio.on('disconnect')
# def handle_disconnect():
#     print('Client disconnected')

# @socketio.on('img_send')
# def handle_img_send(data):
#     try:
#         image_url = data.get('imageBase64')  # Assuming this still contains the URL
#         gender = data.get('gender')
#         if not image_url:
#             raise ValueError('Image URL not found in payload')

#         # Use requests to handle the HTTP request
#         response = requests.get(image_url)
#         if response.status_code == 200:
#             # Convert image data to base64
#             image_base64 = base64.b64encode(response.content)

#             # Decode the base64 to binary for saving to file
#             image_binary = base64.b64decode(image_base64)

#             # Generate a unique filename based on current timestamp
#             image_name = f"{int(time.time() * 1000)}.jpg"

#             # Determine the target directory based on gender
#             if gender == 'male':
#                 target_dir = 'target_images'
#             elif gender == 'female':
#                 target_dir = 'target_images2'
#             else:
#                 raise ValueError('Invalid gender provided')

#             image_path = os.path.join(target_dir, image_name)

#             # Save the image to the file
#             with open(image_path, 'wb') as f:
#                 f.write(image_binary)

#             print('Image saved:', image_path)
#         else:
#             raise Exception(f"Failed to download image, status code {response.status_code}")
#     except Exception as e:
#         print('Error saving image:', str(e))

# if __name__ == '__main__':
    # socketio.run(app, host='192.168.1.226', port=3001)
