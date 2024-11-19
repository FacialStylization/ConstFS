import os
import shutil
import argparse


def organize_and_rename_images(source_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith((".jpg", ".png")):
            name, ext = os.path.splitext(filename)

            parts = name.split("_")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                folder_name = parts[1]
                new_name = parts[0]

                target_folder = os.path.join(source_folder, folder_name)

                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)

                new_file_path = os.path.join(target_folder, new_name + ext)

                old_file_path = os.path.join(source_folder, filename)

                shutil.move(old_file_path, new_file_path)
                print(f'Moved and renamed "{filename}" to "{new_file_path}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize and rename images")
    parser.add_argument("--folder_path", type=str, help="Path to source folder")
    args = parser.parse_args()

    organize_and_rename_images(args.folder_path)
