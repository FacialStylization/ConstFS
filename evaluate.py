import argparse
import os
import lpips
from torchvision import transforms
from PIL import Image

def evaluate(style_images_folder, content_images_folder, stylized_images_folder):
    # Evaluate art_fid
    art_fid_command = f"python -m art_fid --style_images {style_images_folder} --content_images {content_images_folder} --stylized_images {stylized_images_folder}"
    os.system(art_fid_command)

    # Evaluate LPIPS
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

    stylized_images = os.listdir(stylized_images_folder)
    lpips_scores = 0
    for stylized in stylized_images:
        style_image_name = stylized.split('.')[0]
        style_image_path = os.path.join(style_images_folder, style_image_name + '.png')
        stylized_image_path = os.path.join(stylized_images_folder, stylized)

        style_img = Image.open(style_image_path).convert('RGB')
        stylized_img = Image.open(stylized_image_path).convert('RGB')

        style_tensor = transforms.ToTensor()(style_img).unsqueeze(0)
        stylized_tensor = transforms.ToTensor()(stylized_img).unsqueeze(0)

        style_tensor = style_tensor * 2 - 1
        stylized_tensor = stylized_tensor * 2 - 1

        d_alex = loss_fn_alex(style_tensor, stylized_tensor)
        lpips_score = d_alex.item()
        lpips_scores += lpips_score
    
    lpips_scores /= len(stylized_images)
    print(f"LPIPS score: {lpips_scores}")

    # Evaluate pytorch_fid
    fid_score_command = f"python -m pytorch_fid {style_images_folder} {stylized_images_folder} --dims 64"
    os.system(fid_score_command)

    # Evaluate clip_score
    clip_score_command = f"python -m clip_score {style_images_folder} {stylized_images_folder} --real_flag img --fake_flag img"
    os.system(clip_score_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate style transfer results')
    parser.add_argument('--style_images', type=str, help='Path to style images folder')
    parser.add_argument('--content_images', type=str, help='Path to content images folder')
    parser.add_argument('--stylized_images', type=str, help='Path to stylized images folder')
    args = parser.parse_args()

    evaluate(args.style_images, args.content_images, args.stylized_images)
