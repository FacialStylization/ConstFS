import os
import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys


class CurricularFace:
    def __init__(self, model_name="damo/cv_ir101_facerecognition_cfglint"):
        self.model = pipeline(Tasks.face_recognition, model=model_name)

    def get_face_embedding(self, image_path):
        return self.model(image_path)[OutputKeys.IMG_EMBEDDING]

    def calculate_curricular_face_dist(self, reference_path, generated_path):
        ref_embedding = self.get_face_embedding(reference_path)
        gen_embedding = self.get_face_embedding(generated_path)
        return np.dot(ref_embedding[0], gen_embedding[0])

    def calculate_folder_curricular_face_dist(self, reference_folder, generated_folder):
        reference_images = sorted(os.listdir(reference_folder))
        generated_images = sorted(os.listdir(generated_folder))
        distances = []
        for ref_img, gen_img in zip(reference_images, generated_images):
            ref_path = os.path.join(reference_folder, ref_img)
            gen_path = os.path.join(generated_folder, gen_img)
            dist = self.calculate_curricular_face_dist(ref_path, gen_path)
            distances.append(dist)
        print(np.mean(distances))
