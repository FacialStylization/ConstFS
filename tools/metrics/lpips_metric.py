import os
import lpips
import argparse

from torchvision import transforms
from PIL import Image


def evaluate(content_images_folder, stylized_images_folder):
    loss_fn_alex = lpips.LPIPS(net="alex")  # best forward scores

    stylized_images = os.listdir(stylized_images_folder)

    lpips_scores = 0
    for stylized in stylized_images:
        content_image_name = stylized.split(".")[0]
        content_image_path = os.path.join(
            content_images_folder, content_image_name + ".jpg"
        )
        stylized_image_path = os.path.join(stylized_images_folder, stylized)

        content_img = Image.open(content_image_path).convert("RGB")
        stylized_img = Image.open(stylized_image_path).convert("RGB")

        content_tensor = transforms.ToTensor()(content_img).unsqueeze(0)
        stylized_tensor = transforms.ToTensor()(stylized_img).unsqueeze(0)

        content_tensor = content_tensor * 2 - 1
        stylized_tensor = stylized_tensor * 2 - 1

        d_alex = loss_fn_alex(content_tensor, stylized_tensor)
        lpips_score = d_alex.item()
        lpips_scores += lpips_score
    lpips_scores /= len(stylized_images)
    print(f"LPIPS score: {lpips_scores}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate style transfer results")
    parser.add_argument(
        "--content_images", type=str, help="Path to content images folder"
    )
    parser.add_argument(
        "--stylized_images", type=str, help="Path to stylized images folder"
    )
    args = parser.parse_args()

    evaluate(args.content_images, args.stylized_images)
