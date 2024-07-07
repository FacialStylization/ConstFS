import sys
sys.path.append('./')

import os 
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, UNet2DConditionModel, StableDiffusionXLControlNetPipeline
from safetensors.torch import load_file

# import spaces
import gradio as gr

from pipeline import IPAdapterXL

import os
# os.system("git lfs install")
# os.system("git clone https://huggingface.co/h94/IP-Adapter")
# os.system("mv IP-Adapter/sdxl_models sdxl_models")

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# initialization
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "IP-Adapter/sdxl_models/image_encoder"
ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
config_path = "models/unet_config.json"
unet_path = "models/sdxl_lightning_4step_unet.safetensors"
controlnet_path = "models/canny"
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)

# config = UNet2DConditionModel.load_config(config_path)
# unet = UNet2DConditionModel.from_config(config).to(device, torch.float16)
# unet.load_state_dict(load_file(unet_path, device=device))

# load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

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

def generate_target_blocks(style_input_str, layout_input_str, is_style_only):
    up_blocks_str = "up_blocks"
    down_blocks_str = "down_blocks"
    attention_str = "attentions.1"
    
    if style_input_str.isdigit():
        style_target_block = up_blocks_str + '.' + style_input_str + '.' + attention_str
        style_target_blocks = [style_target_block]
        if is_style_only:
            return style_target_blocks
        else:
            if layout_input_str.isdigit():
                layout_target_block = down_blocks_str + '.' + layout_input_str + '.' + attention_str
                style_target_blocks.append(layout_target_block)
            else:
                layout_target_blocks = [down_blocks_str + '.' + num.strip() + '.' + attention_str for num in layout_input_str.split(',')]
                style_target_blocks.extend(layout_target_blocks)
            return style_target_blocks
    else:
        style_target_blocks = [up_blocks_str + '.' + num.strip() + '.' + attention_str for num in style_input_str.split(',')]
        if is_style_only:
            return style_target_blocks
        if layout_input_str.isdigit():
            layout_target_block = down_blocks_str + '.' + layout_input_str + '.' + attention_str
            style_target_blocks.append(layout_target_block)
        else:
            layout_target_blocks = [down_blocks_str + '.' + num.strip() + '.' + attention_str for num in layout_input_str.split(',')]
            style_target_blocks.extend(layout_target_blocks)
        return style_target_blocks

# @spaces.GPU(enable_queue=True)
def create_image(image_pil,
                 input_image,
                 prompt,
                 n_prompt,
                 scale, 
                 control_scale, 
                 guidance_scale,
                 num_samples,
                 num_inference_steps,
                 seed,
                 style_blocks="0",
                 layout_blocks="2",
                 target="Load only style blocks",
                 neg_content_prompt=None,
                 neg_content_scale=0):

    if target =="Load original IP-Adapter":
        # target_blocks=["blocks"] for original IP-Adapter
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["blocks"])
    elif target=="Load only style blocks":
        # target_blocks=["up_blocks.0.attentions.1"] for style blocks only
        target_blocks_gen = generate_target_blocks(style_blocks, layout_blocks, True)
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=target_blocks_gen)
    elif target == "Load style+layout block":
        # target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
        target_blocks_gen = generate_target_blocks(style_blocks, layout_blocks, False)
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=target_blocks_gen)
    
    if input_image is not None:
        input_image = resize_img(input_image, max_side=1024)
        cv_input_image = pil_to_cv2(input_image)
        detected_map = cv2.Canny(cv_input_image, 50, 200)
        canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
    else:
        canny_map = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
        control_scale = 0

    if float(control_scale) == 0:
        canny_map = canny_map.resize((1024,1024))
    
    if len(neg_content_prompt) > 0 and neg_content_scale != 0:
        images = ip_model.generate(pil_image=image_pil,
                                prompt=prompt,
                                negative_prompt=n_prompt,
                                scale=scale,
                                guidance_scale=guidance_scale,
                                num_samples=num_samples,
                                num_inference_steps=num_inference_steps, 
                                seed=seed,
                                pulid_image=cv_input_image,
                                image=canny_map,
                                controlnet_conditioning_scale=float(control_scale),
                                neg_content_prompt=neg_content_prompt,
                                neg_content_scale=neg_content_scale
                                )
    else:
        images = ip_model.generate(pil_image=image_pil,
                                prompt=prompt,
                                negative_prompt=n_prompt,
                                scale=scale,
                                guidance_scale=guidance_scale,
                                num_samples=num_samples,
                                num_inference_steps=num_inference_steps, 
                                seed=seed,
                                pulid_image=cv_input_image,
                                image=canny_map,
                                controlnet_conditioning_scale=float(control_scale),
                                )
    return images

def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2

# Description
title = r"""
<h1 align="center">Instant Facial Stylization Experiment</h1>
"""


block = gr.Blocks(css="footer {visibility: hidden}").queue(max_size=10, api_open=False)
with block:
    
    # description
    gr.Markdown(title)
    
    with gr.Tabs():
        with gr.Row():
            with gr.Column():
                with gr.Column():
                    src_image_pil = gr.Image(label="Source Image (optional)", type='pil')
                    control_scale = gr.Slider(minimum=0,maximum=1.0, step=0.01,value=0.5, label="Controlnet conditioning scale")
                    
                    n_prompt = gr.Textbox(label="Neg Prompt", value="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry")
                    
                    neg_content_prompt = gr.Textbox(label="Neg Content Prompt", value="")
                    neg_content_scale = gr.Slider(minimum=0, maximum=1.0, step=0.01,value=0.5, label="Neg Content Scale")

                    guidance_scale = gr.Slider(minimum=1,maximum=15.0, step=0.01,value=5.0, label="guidance scale")
                    num_samples= gr.Slider(minimum=1,maximum=4.0, step=1.0,value=1.0, label="num samples")
                    num_inference_steps = gr.Slider(minimum=5,maximum=50.0, step=1.0,value=20, label="num inference steps")
                    seed = gr.Slider(minimum=-1000000,maximum=1000000,value=1, step=1, label="Seed Value")
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Column():
                
                with gr.Row():
                    with gr.Column():
                        image_pil = gr.Image(label="Style Image", type='pil')
                
                target = gr.Radio(["Load only style blocks", "Load style+layout block", "Load original IP-Adapter"], 
                                  value="Load style+layout block",
                                  label="Style mode")
                
                prompt = gr.Textbox(label="Prompt",
                                    value="masterpiece, best quality, high quality")
                
                style_blocks = gr.Textbox(label="Style Blocks", value="0")

                layout_blocks = gr.Textbox(label="Layout Blocks", value="2")

                scale = gr.Slider(minimum=0,maximum=2.0, step=0.01,value=1.0, label="Scale")
                
                generate_button = gr.Button("Generate Image")
                
                generated_image = gr.Gallery(label="Generated Image")

        generate_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=create_image,
            inputs=[image_pil,
                    src_image_pil,
                    prompt,
                    n_prompt,
                    scale, 
                    control_scale, 
                    guidance_scale,
                    num_samples,
                    num_inference_steps,
                    seed,
                    style_blocks,
                    layout_blocks,
                    target,
                    neg_content_prompt,
                    neg_content_scale], 
            outputs=[generated_image])

block.launch(server_port=6006)
