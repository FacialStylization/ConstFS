import os
import random
import shutil

current_directory = os.getcwd()

new_folder = os.path.join(current_directory, os.path.basename(current_directory))
os.makedirs(new_folder, exist_ok=True)

files = os.listdir(current_directory)

image_files = [file for file in files if file.endswith((".jpg", ".jpeg", ".png"))]

selected_images = random.sample(image_files, 100)

for image in selected_images:
    source = os.path.join(current_directory, image)
    destination = os.path.join(new_folder, image)
    shutil.move(source, destination)

print("Images have been randomly selected and moved to the new folder.")
