import requests
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch

import transformers
from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTFeatureExtractor

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# pad image to square size
def preprocess_image(pil_image, px_from_border=30):
    image = np.array(pil_image)
        
    if image.shape[0]<image.shape[1]:
        image_cut = image[-px_from_border:, :, :]
        mean_side_color = image_cut.reshape(-1, 3).mean(0).astype(np.uint8).tolist()
        image_padded = cv2.copyMakeBorder(image.copy(), 0, image.shape[1]-image.shape[0], 0, 0, cv2.BORDER_CONSTANT, 
                                          value=mean_side_color)
    else:
        image_cut = image[:, -px_from_border:, :]
        mean_side_color = image_cut.reshape(-1, 3).mean(0).astype(np.uint8).tolist()
        image_padded = cv2.copyMakeBorder(image.copy(), 0, 0, 0, image.shape[0]-image.shape[1], cv2.BORDER_CONSTANT, 
                                          value=mean_side_color)
    return image_padded


class OWLViTModel:
    def __init__(self, labels, cache_dir, device='cuda:0', score_threshold=0.1):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir=cache_dir)
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir=cache_dir)
        self.model.eval()
        self.model.to(device)
        
        self.cache_dir = cache_dir
        self.device = device
        
        self.labels = [labels]
        self.score_threshold = score_threshold
        
        #
        inputs = self.processor(text=self.labels, return_tensors="pt")
        self.input_ids = inputs['input_ids'].to(self.device)
        self.attention_mask = inputs['attention_mask'].to(self.device)
        
    def _get_boxes_batched(self, inputs, images):
        bs = len(images)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
        #
        input_ids = torch.cat([self.input_ids]*bs).to(self.device)
        attention_mask = torch.cat([self.attention_mask]*bs).to(self.device)
        
        inputs['input_ids'] = input_ids
        inputs['attention_mask'] = attention_mask
        #
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([im.size[::-1] for im in images]).to(self.device)
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)
        
        all_samples_data = []
        for i in range(len(results)):
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            boxes = boxes.cpu().numpy().astype(np.int32)

            sample_data = []
            for box, score, label in zip(boxes, scores, labels):
                if score >= self.score_threshold:
                    box = [i for i in box.tolist()]
                    class_label = self.labels[0][label]
                    confidence = round(score.item(), 3)
                    sample_data.append({"label": class_label, "confidence": confidence, "box": box})
            all_samples_data.append(sample_data)
        return all_samples_data
        
    def get_boxes(self, pil_image):
        img_preproc = Image.fromarray(preprocess_image(pil_image))
        images = [img_preproc]
        inputs = self.processor(images=[img_preproc], return_tensors="pt")
        return self._get_boxes_batched(inputs, images)[0]