import dashscope
import random
from http import HTTPStatus
from dashscope import Generation

dashscope.api_key ="sk-779e676f1ba1477aadae1a37f1a20027"

class ImageCaptioner:
    
    def __init__(self, style_image_pil, content_image_pil):
        self.style_image_pil = style_image_pil
        self.content_image_pil = content_image_pil

    def post_process_prompt(self, raw_prompt):
        tags = [tag.strip().lower() for tag in raw_prompt.split(',') if tag.strip()]

        tags = ['_'.join(tag.split()) for tag in tags]
        
        seen = set()
        unique_tags = [tag for tag in tags if not (tag in seen or seen.add(tag))]
        
        final_tags = unique_tags[:70]
        
        return ', '.join(final_tags)
    
    def get_image_captions(self, image, user_prompt):
        """Simple single round multimodal conversation call.
        """
        messages = [
            {
                "role": "system", 
                "content": "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image."
            },
            {
                "role": "user",
                "content": [
                    {"image": f"{image}"},
                    {"text": f"{user_prompt}"}
                ]
            }
        ]
        
        response = dashscope.MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages
        )
        
        # The response status_code is HTTPStatus.OK indicate success,
        # otherwise indicate request is failed, you can get error code
        # and message from code and message.
        if response.status_code == HTTPStatus.OK:
            print(response)
            return self.get_response_content(response)
        else:
            print(response.code)  # The error code.
            print(response.message)  # The error message.

    def get_response_content(self, response):
        return response.output.choices[0].message.content[0]["text"]
    
    def get_integrated_prompt(self):
            messages = [
                {
                    "role": "system", 
                    "content": "You are an AI system specializing in style transfer image analysis. Generate a single, cohesive prompt that combines stylized and realistic elements for facial images. Use lowercase words, connecting phrases with underscores. Balance stylized features with realistic details. Focus on visual descriptors useful for image recreation."
                },
                {
                    "role": "user", 
                    "content": f"Combine these style and content prompts into a single, integrated prompt for a style-transferred face image:\nStyle: {self.style_prompt}\nContent: {self.content_prompt}\nGenerate 20-50 comma-separated tags. Include artistic style, character design, facial features, color, lighting, gender, age, ethnicity, hair, eyes, expression, accessories, and makeup. Avoid repetition and abstract concepts."
                }
            ]
            
            response = Generation.call(model="qwen2-72b-instruct",
                                    messages=messages,
                                    seed=random.randint(1, 10000),
                                    result_format="message")
            
            if response.status_code == HTTPStatus.OK:
                raw_prompt = response.output.choices[0].message.content
                processed_prompt = self.post_process_prompt(raw_prompt)
                return processed_prompt
            else:
                print(f"Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}")
                return None

    def generate_prompt(self):
        self.style_prompt = self.get_image_captions(self.style_image_pil, "Image tagging expert, provide precise tags for this stylized face image. Use concise keywords. Focus: Artistic style, Character design, Facial features, Color palette, Lighting, Stylistic choices. Include: Gender, age, hair, eyes, facial features, expression, accessories. Identify any recognizable characters/IPs. 20-75 accurate, non-repetitive tags, comma-separated.")
        self.content_prompt = self.get_image_captions(self.content_image_pil, "Image tagging expert, provide precise tags for this realistic face image. Use concise keywords. Focus: Facial features, Skin tone, Expression, Hair, Lighting, Image quality. Include: Gender, age, ethnicity, eyes, facial features, expression, accessories, makeup, portrait style. Identify any celebrities. 20-75 accurate, non-repetitive tags, comma-separated.")
        
        prompt = self.get_integrated_prompt()
        return prompt
