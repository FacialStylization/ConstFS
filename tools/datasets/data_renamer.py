import os

current_directory = os.getcwd()

folder_path = os.path.join(current_directory, os.path.basename(current_directory))

file_list = os.listdir(folder_path)

file_list.sort()

for i, file_name in enumerate(file_list):
    new_file_name = str(i + 1) + os.path.splitext(file_name)[1]

    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_file_name)

    os.rename(old_file_path, new_file_path)
