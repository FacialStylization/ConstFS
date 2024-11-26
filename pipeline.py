from pulid.pipeline import PuLIDPipeline
from ip_adapter.pipeline import IPAdapter
from PIL import Image
from typing import List
import torch

pipeline = PuLIDPipeline()

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator

def split_tiles(embeds, num_split):
    if len(embeds.shape) == 2:
        batch_size, channels = embeds.shape
        height = width = int(channels ** 0.5)
        embeds = embeds.view(batch_size, height, width, -1)
        print(f"Adjusted embeds shape: {embeds.shape}")

    _, H, W, _ = embeds.shape
    out = []
    for x in embeds:
        x = x.unsqueeze(0)
        h, w = H // num_split, W // num_split
        x_split = torch.cat([x[:, i*h:(i+1)*h, j*w:(j+1)*w, :] for i in range(num_split) for j in range(num_split)], dim=0)    
        out.append(x_split)
        
    x_split = torch.stack(out, dim=0)
        
    return x_split
        
def merge_embeddings(x, tiles):
    chunk_size = tiles * tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        grid_size = int(num_tiles ** 0.5)
        tile_size = int(embeds.shape[1] ** 0.5)
        
        # Reshape to [grid_size, grid_size, tile_size, tile_size]
        reshaped = embeds.reshape(grid_size, grid_size, tile_size, tile_size)
        
        # Merge the tiles
        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                            for i in range(grid_size)], dim=0)
        
        # Flatten to [1, grid_size * tile_size * grid_size * tile_size]
        merged = merged.view(1, -1)
        
        out.append(merged)
    
    out = torch.cat(out, dim=0)
    
    return out

class IPAdapterXL(IPAdapter):

    """SDXL"""

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, content_prompt_embeds=None, tiles=4):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image).image_embeds
            print(f"first clip_image_embeds shape: {clip_image_embeds.shape}")
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
        if content_prompt_embeds is not None:
            clip_image_embeds = clip_image_embeds - content_prompt_embeds.to(self.device, dtype=torch.float16)
            print(f"second clip_image_embeds shape: {clip_image_embeds.shape}")

    
        if tiles > 1:
            # Split in tiles
            image_split = split_tiles(clip_image_embeds, tiles)

            # Get the embeds for each tile
            embeds_split = {"image_embeds": [], "penultimate_hidden_states": []}
            for tile in image_split:
                encoded = self.image_encoder(tile, output_hidden_states=True)
                embeds_split["image_embeds"].append(encoded.image_embeds)
                embeds_split["penultimate_hidden_states"].append(encoded.hidden_states[-2])

                # Concatenate the embeddings
                embeds_split["image_embeds"] = torch.cat(embeds_split["image_embeds"], dim=0)
                embeds_split["penultimate_hidden_states"] = torch.cat(embeds_split["penultimate_hidden_states"], dim=0)

                # Merge the embeddings
                embeds_split["image_embeds"] = merge_embeddings(embeds_split["image_embeds"], tiles)
                embeds_split["penultimate_hidden_states"] = merge_embeddings(embeds_split["penultimate_hidden_states"], tiles)

                # Update the clip_image_embeds
                clip_image_embeds = embeds_split["image_embeds"]
                print(f"third clip_image_embeds shape: {clip_image_embeds.shape}")

        print(f"last clip_image_embeds shape: {clip_image_embeds.shape}")

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        neg_content_emb=None,
        neg_content_prompt=None,
        neg_content_scale=1.0,
        **kwargs,
    ):  
        # print('kwargs', kwargs)
        image = kwargs['pulid_image']
        kwargs.pop('pulid_image')
        id_embedding = pipeline.get_id_embedding(image)
        
        self.set_scale(scale)

        # Set the number of prompts based on the number of images: 
        # 1 if there's a single image, or the number of images if there's a list of images.
        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        # Handle the generation and assignment of negative content embeddings. 
        # If `neg_content_emb` is None and `neg_content_prompt` is not None, 
        # encode the negative content prompt and scale the pooled prompt embeddings. 
        # If `neg_content_emb` is not None, assign None to `pooled_prompt_embeds_`.
        if neg_content_emb is None:
            if neg_content_prompt is not None:
                with torch.inference_mode():
                    (
                        prompt_embeds_, # torch.Size([1, 77, 2048])
                        negative_prompt_embeds_,
                        pooled_prompt_embeds_, # torch.Size([1, 1280])
                        negative_pooled_prompt_embeds_,
                    ) = self.pipe.encode_prompt(
                        neg_content_prompt,
                        num_images_per_prompt=num_samples,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    pooled_prompt_embeds_ *= neg_content_scale
            else:
                pooled_prompt_embeds_ = neg_content_emb # same as None
        else:
            pooled_prompt_embeds_ = None

        # Get the image prompt embeddings and unconditional image prompt embeddings
        # by calling the `get_image_embeds` method with the PIL image and the pooled prompt embeddings.
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image, content_prompt_embeds=pooled_prompt_embeds_)
        # print('pooled_prompt_embeds_', pooled_prompt_embeds_)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            id_embedding=id_embedding,
            **kwargs,
        ).images
    
        return images
        