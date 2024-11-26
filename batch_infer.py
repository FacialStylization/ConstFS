import os
import cv2
import torch
import argparse
import numpy as np

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image

from pipeline import IPAdapterXL
from utils.image_captioner import ImageCaptioner

class StyleTransfer:
    def __init__(self, style_folder_path, content_folder_path):
        self.style_folder_path = style_folder_path
        self.content_folder_path = content_folder_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path = "IP-Adapter/sdxl_models/image_encoder"
        ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter_sdxl.safetensors"
        controlnet_path = "models/canny"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, use_safetensors=True, torch_dtype=torch.float16
        ).to(self.device)
        # load SDXL pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
        self.pipe.enable_vae_tiling()

        # load ip-adapter
        # target_blocks=["block"] for original IP-Adapter
        # target_blocks=["up_blocks.0.attentions.1"] for style blocks only
        target_blocks = [
            "up_blocks.0.attentions.1",
            "down_blocks.2.attentions.1",
        ]  # for style+layout blocks
        self.ip_model = IPAdapterXL(
            self.pipe,
            image_encoder_path,
            ip_ckpt,
            self.device,
            num_tokens=4,
            target_blocks=target_blocks,
        )

    def resize_img(
        self,
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

    def pil_to_cv2(self, image_pil):
        image_np = np.array(image_pil)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_cv2

    def create_prompt(self, style_image_path, content_image_path):
        image_captioner = ImageCaptioner(style_image_path, content_image_path)
        prompt = image_captioner.generate_prompt()
        return prompt
    
    def generate(self):
        style_files = os.listdir(self.style_folder_path)
        for style_file in style_files:
            if not style_file.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            style_path = os.path.join(self.style_folder_path, style_file)
            content_file = style_file.split(".")[0] + ".jpg"
            content_path = os.path.join(self.content_folder_path, content_file)

            style_image = Image.open(style_path)
            style_image = self.resize_img(style_image, max_side=1024)

            content_image = Image.open(content_path)
            content_image = self.resize_img(content_image, max_side=1024)
            cv_input_image = self.pil_to_cv2(content_image)
            detected_map = cv2.Canny(cv_input_image, 50, 200)
            canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

            prompt = create_prompt(style_path, content_path)
            # generate image
            images = self.ip_model.generate(
                pil_image=style_image,
                prompt=prompt,
                negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                scale=1.0,
                guidance_scale=5,
                num_samples=1,
                num_inference_steps=20,
                seed=42,
                pulid_image=cv_input_image,
                image=canny_map,
                controlnet_conditioning_scale=0.5,
            )

            file_name = style_file.split(".")[0]
            output_path = f"results/{file_name}.png"
            images[0].save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style transfer")
    parser.add_argument(
        "--style_folder_path", type=str, help="Path to the style folder"
    )
    parser.add_argument(
        "--content_folder_path", type=str, help="Path to the content folder"
    )
    args = parser.parse_args()

    style_transfer = StyleTransfer(args.style_folder_path, args.content_folder_path)
    style_transfer.generate()
