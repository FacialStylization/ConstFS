import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from insightface.app import FaceAnalysis
import insightface
import numpy as np


class ArcFaceMetric:
    def __init__(self, model_name="buffalo_l", ctx_id=0):
        # 初始化 ArcFace 模型
        self.app = FaceAnalysis(model_name)
        self.app.prepare(ctx_id=ctx_id)

        # 图像预处理
        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def get_face_embedding(self, image_path):
        # 读取图像并转换为 NumPy 数组
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)

        # 获取人脸嵌入
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        return faces[0].embedding

    def calculate_arcface_dist(self, reference_path, generated_path):
        ref_embedding = self.get_face_embedding(reference_path)
        gen_embedding = self.get_face_embedding(generated_path)
        if ref_embedding is None or gen_embedding is None:
            return None
        return (
            1
            - F.cosine_similarity(
                torch.tensor(ref_embedding), torch.tensor(gen_embedding), dim=0
            ).item()
        )

    def evaluate_folders(self, reference_folder, generated_folder):
        reference_images = sorted(os.listdir(reference_folder))
        generated_images = sorted(os.listdir(generated_folder))
        distances = []
        for ref_img, gen_img in zip(reference_images, generated_images):
            ref_path = os.path.join(reference_folder, ref_img)
            gen_path = os.path.join(generated_folder, gen_img)
            dist = self.calculate_arcface_dist(ref_path, gen_path)
            if dist is not None:
                distances.append(dist)
        print(f"ArcFace distance: {np.mean(distances)}")
