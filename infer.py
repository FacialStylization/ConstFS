import cv2
import torch
import argparse
import numpy as np

from diffusers import StableDiffusionXLPipeline, ControlNetModel
from PIL import Image

from ip_adapter import IPAdapterPlusXL

def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[
            offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new
        ] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2

def style_transfer(style_path, content_path):
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    image_encoder_path = "models/image_encoder"
    ip_ckpt = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
    device = "cuda"
    controlnet_path = "models/canny"
    controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)
    # load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()

    # load ip-adapter
    # target_blocks=["block"] for original IP-Adapter
    # target_blocks=["up_blocks.0.attentions.1"] for style blocks only
    target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
    ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16, target_blocks=target_blocks)

    image = Image.open(content_path)
    image = resize_img(image, max_side=1024)
    cv_input_image = pil_to_cv2(image)
    detected_map = cv2.Canny(cv_input_image, 50, 200)
    canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

    # generate image
    images = ip_model.generate(pil_image=image,
                            prompt="a cat, masterpiece, best quality, high quality",
                            negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                            scale=1.0,
                            guidance_scale=5,
                            num_samples=1,
                            num_inference_steps=30, 
                            seed=42,
                            pulid_image=cv_input_image,
                            image=canny_map,
                            controlnet_conditioning_scale=0.5,
                            )
    
    file_name = style_path.split("/")[-1].split(".")[0]
    output_path = f"results/{file_name}.png"
    images[0].save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Style transfer')
    parser.add_argument('--style_path', type=str, help='Path to the style image')
    parser.add_argument('--content_path', type=str, help='Path to the content image')
    args = parser.parse_args()

    style_transfer(args.style_path, args.content_path)
