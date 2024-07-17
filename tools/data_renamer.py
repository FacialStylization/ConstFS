import os

# Get the current working directory
current_directory = os.getcwd()

# Construct the folder path
folder_path = os.path.join(current_directory, os.path.basename(current_directory))

# Get the list of files in the folder
file_list = os.listdir(folder_path)

# Sort the file list alphabetically
file_list.sort()

# Iterate over the files and rename them
for i, file_name in enumerate(file_list):
    # Generate the new file name
    new_file_name = str(i + 1) + os.path.splitext(file_name)[1]

    # Construct the full path of the file
    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_file_name)

    # Rename the file
    os.rename(old_file_path, new_file_path)
