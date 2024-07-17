import os
import random
import shutil

# Get the current directory
current_directory = os.getcwd()

# Create a new folder with the name of the current directory
new_folder = os.path.join(current_directory, os.path.basename(current_directory))
os.makedirs(new_folder, exist_ok=True)

# Get a list of all files in the current directory
files = os.listdir(current_directory)

# Filter out non-image files
image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# Randomly select 100 images
selected_images = random.sample(image_files, 100)

# Move the selected images to the new folder
for image in selected_images:
    source = os.path.join(current_directory, image)
    destination = os.path.join(new_folder, image)
    shutil.move(source, destination)

print("Images have been randomly selected and moved to the new folder.")
