import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from insightface.app import FaceAnalysis

class ArcFaceMetric:
    def __init__(self, model_name='buffalo_l', ctx_id=0):
        # 初始化 ArcFace 模型
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id)  # 使用 GPU (ctx_id=0)，使用 CPU 时设为 ctx_id=-1

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),  # ArcFace 输入通常为 112x112
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0)

    def get_face_embedding(self, image_path):
        img = Image.open(image_path).convert("RGB")
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("未检测到人脸，请检查图像质量或更换图像。")
        embedding = faces[0].embedding
        return torch.tensor(embedding).unsqueeze(0)

    def calculate_arcface_dist(self, reference_path, generated_path):
        ref_embedding = self.get_face_embedding(reference_path)
        gen_embedding = self.get_face_embedding(generated_path)

        # 标准化特征向量
        ref_embedding = F.normalize(ref_embedding)
        gen_embedding = F.normalize(gen_embedding)

        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(ref_embedding, gen_embedding)
        arcface_dist = 1 - cosine_sim.item()  # Arcface-Dist 值越小越好

        return arcface_dist

    def evaluate_folders(self, reference_folder, generated_folder):
        reference_images = sorted(os.listdir(reference_folder))
        generated_images = sorted(os.listdir(generated_folder))

        if len(reference_images) != len(generated_images):
            raise ValueError("两个文件夹中的图像数量不匹配。")

        distances = {}
        for ref_img, gen_img in zip(reference_images, generated_images):
            ref_path = os.path.join(reference_folder, ref_img)
            gen_path = os.path.join(generated_folder, gen_img)
            dist = self.calculate_arcface_dist(ref_path, gen_path)
            distances[ref_img] = dist
        
        # 计算平均 Arcface-Dist
        avg_dist = sum(distances.values()) / len(distances)
        print(f"Arcface-Dist = {avg_dist:.4f}")

        return distances
