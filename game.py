# import os
# import pygame
# from pygame.locals import *
# import time
# import threading


# RESULT_FOLDER = 'results'
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# # def display_image(image_path):
# #     pygame.init()
# #     screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.NOFRAME)
# #     pygame.display.set_caption('Full Screen Image Viewer')

# #     # Load and play the video
# #     video_path = 'intro_video.mp4'  # Change this to the path of your video file
# #     pygame.mixer.quit()  # Stop any audio playback
# #     pygame.movie.Movie(video_path).play()

# #     # Wait for 5 seconds to play the video
# #     start_time = time.time()
# #     while time.time() - start_time < 5:
# #         for event in pygame.event.get():
# #             if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
# #                 pygame.quit()
# #                 return

# #     # Load and display the image after the video
# #     image = pygame.image.load(image_path)
# #     screen_width, screen_height = screen.get_size()
# #     image_width, image_height = image.get_size()

# #     # Calculate the position to center the image vertically
# #     x = (screen_width - image_width) // 2
# #     y = (screen_height - image_height) // 2

# #     screen.blit(image, (x, y))
# #     pygame.display.flip()

# #     # Display image for 7 seconds
# #     start_time = time.time()
# #     while time.time() - start_time < 7:
# #         for event in pygame.event.get():
# #             if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
# #                 pygame.quit()
# #                 return
# #     pygame.quit()

# # def display_image(image_path):
# #     pygame.init()
# #     screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.NOFRAME)
# #     pygame.display.set_caption('Full Screen Image Viewer')

# #     # Load the video
# #     video_path = 'vid.mp4'  # Change this to the path of your video file
# #     pygame.mixer.quit()  # Stop any audio playback
# #     pygame.mixer.init()  # Initialize the mixer for video playback
# #     pygame.mixer.music.load(video_path)
    
# #     # Start the video playback
# #     pygame.mixer.music.play()

# #     # Wait for the video to play for 5 seconds
# #     start_time = time.time()
# #     while time.time() - start_time < 5 and pygame.mixer.music.get_busy():
# #         for event in pygame.event.get():
# #             if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
# #                 pygame.quit()
# #                 return

# #     # Load and display the image after the video
# #     image = pygame.image.load(image_path)
# #     screen_width, screen_height = screen.get_size()
# #     image_width, image_height = image.get_size()

# #     # Calculate the position to center the image vertically
# #     x = (screen_width - image_width) // 2
# #     y = (screen_height - image_height) // 2

# #     screen.blit(image, (x, y))
# #     pygame.display.flip()

# #     # Display image for 7 seconds
# #     start_time = time.time()
# #     while time.time() - start_time < 7:
# #         for event in pygame.event.get():
# #             if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
# #                 pygame.quit()
# #                 return
# #     pygame.quit()

# def display_image(image_path):
#     pygame.init()
#     screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.NOFRAME)
#     pygame.display.set_caption('Full Screen Image Viewer')
#     image = pygame.image.load(image_path)

#     screen_width, screen_height = screen.get_size()
#     image_width, image_height = image.get_size()

#     # Calculate the position to center the image vertically
#     x = (screen_width - image_width) // 2
#     y = (screen_height - image_height) // 2

#     screen.blit(image, (x, y))
#     pygame.display.flip()

#     # Display image for 7 seconds
#     start_time = time.time()
#     while time.time() - start_time < 7:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
#                 pygame.quit()
#                 return
#     pygame.quit()


# def watch_result_folder():
#     result_before = set(os.listdir(RESULT_FOLDER))
#     while True:
#         result_after = set(os.listdir(RESULT_FOLDER))

#         new_images = result_after - result_before
#         for image in new_images:
#             image_path = os.path.join(RESULT_FOLDER, image)
#             display_image(image_path)

#         result_before = result_after

#         # Check for new images every second
#         time.sleep(1)
        


# # Watch result folder in a separate thread
# watch_thread = threading.Thread(target=watch_result_folder)
# watch_thread.daemon = True
# watch_thread.start()

# # Ensure that the watch thread keeps running
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     pass










import os
import pygame
import time
import threading
from pygame.locals import *

RESULT_FOLDER = 'results'
os.makedirs(RESULT_FOLDER, exist_ok=True)

def play_video(video_path):
    try:
        pygame.init()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.NOFRAME)
        pygame.display.set_caption('Video Player')

        # Load the video as a surface
        video_surface = pygame.image.load(video_path)

        # Blit the video surface to the screen
        screen.blit(video_surface, (0, 0))
        pygame.display.flip()

        # Play the video for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return

        pygame.quit()
    except Exception as e:
        print(f"Error playing video: {e}")

def display_image(image_path):
    try:
        pygame.init()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.NOFRAME)
        pygame.display.set_caption('Full Screen Image Viewer')

        # Play video for 5 seconds before displaying the image
        play_video('vid.mp4')   

        # Load the image
        image = pygame.image.load(image_path)
        screen_width, screen_height = screen.get_size()
        image_width, image_height = image.get_size()

        # Calculate the position to center the image vertically
        x = (screen_width - image_width) // 2
        y = (screen_height - image_height) // 2

        screen.blit(image, (x, y))
        pygame.display.flip()

        # Display image for 7 seconds
        start_time = time.time()
        while time.time() - start_time < 7:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
        pygame.quit()
    except Exception as e:
        print(f"Error displaying image: {e}")

def watch_result_folder():
    result_before = set(os.listdir(RESULT_FOLDER))
    while True:
        result_after = set(os.listdir(RESULT_FOLDER))
        new_files = result_after - result_before

        for file in new_files:
            file_path = os.path.join(RESULT_FOLDER, file)
            if file.endswith('.jpg') or file.endswith('.png'):
                display_image(file_path)

        result_before = result_after
        time.sleep(1)

# Watch result folder in a separate thread
watch_thread = threading.Thread(target=watch_result_folder)
watch_thread.daemon = True
watch_thread.start()

# Ensure that the watch thread keeps running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
