import argparse
import os
from tools.metrics.arcface_metric_0 import ArcFaceMetric

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate(style_images_folder, content_images_folder, stylized_images_folder):
    # Evaluate art_fid
    art_fid_command = f"python -m art_fid --style_images {style_images_folder} --content_images {content_images_folder} --stylized_images {stylized_images_folder}"
    os.system(art_fid_command)

    # Evaluate clip_score
    clip_score_command = f"python -m clip_score {style_images_folder} {stylized_images_folder} --real_flag img --fake_flag img"
    os.system(clip_score_command)

    # Evaluate arcface_distance
    arc_margin_product = ArcFaceMetric()
    arc_margin_product.evaluate_folders(content_images_folder, stylized_images_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate style transfer results")
    parser.add_argument(
        "--style_images",
        default="datasets/aahq",
        type=str,
        help="Path to style images folder",
    )
    parser.add_argument(
        "--content_images",
        default="datasets/celeba-1024",
        type=str,
        help="Path to content images folder",
    )
    parser.add_argument(
        "--stylized_images", type=str, help="Path to stylized images folder"
    )
    args = parser.parse_args()

    evaluate(args.style_images, args.content_images, args.stylized_images)
