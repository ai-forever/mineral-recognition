import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image
import numpy as np

import cv2
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import json
import zipfile
from shapely.geometry import Polygon

from collections import OrderedDict

from .craft import CRAFT
from .refinenet import RefineNet
import scripts.preprocess.CRAFT.craft_utils as craft_utils
import scripts.preprocess.CRAFT.imgproc as imgproc
import scripts.preprocess.CRAFT.file_utils as file_utils
from scripts.utils.fp16 import FP16Module


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def box_to_poly(box):
    return [box[0], [box[0][0], box[1][1]], box[1], [box[1][0], box[0][1]]]

def boxes_area(img, bboxes):
    img_s = img.shape[0]*img.shape[1]
    total_S = 0
    for box in bboxes:
        pgon = Polygon(box_to_poly(box)) 
        S = pgon.area
        total_S+=S
    return total_S/img_s
    
def preprocess_image(image, canvas_size, mag_ratio):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    return x, ratio_w, ratio_h

# load net
def init_CRAFT_model(chekpoint_path, device):
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(chekpoint_path)))
    net = FP16Module(net)
    net = net.to(device)
    net.eval()
    return net

def init_model_refiner(chekpoint_path, device):
    refine_net = RefineNet()
    refine_net.load_state_dict(copyStateDict(torch.load(chekpoint_path)))
    refine_net = refine_net.to(device)
    refine_net.eval()
    return refine_net

def draw_boxes(image, boxes):
    img = image.copy()
    for i, box in enumerate(boxes):
        poly_ = np.array(box_to_poly(box)).astype(np.int32).reshape((-1))
        poly_ = poly_.reshape(-1, 2)
        cv2.polylines(img, [poly_.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        ptColor = (0, 255, 255)
    return Image.fromarray(img)


class CRAFTModel:
    def __init__(
        self, 
        model_path,
        refiner_path,
        device,
        canvas_size = 1280,
        mag_ratio = 1.5,
        text_threshold = 0.7,
        link_threshold = 0.4,
        low_text = 0.4,
        poly = True
    ):
        self.model_path = model_path
        self.refiner_path = refiner_path
        self.device = device
        
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.poly = poly
        
        self.net = init_CRAFT_model(model_path, device)
        self.refiner = init_model_refiner(refiner_path, device)
        
    def _get_boxes_preprocessed(self, x, ratio_w, ratio_h):
        x = x.to(self.device)

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        with torch.no_grad():
            y_refiner = self.refiner(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, 
            self.text_threshold, self.link_threshold, 
            self.low_text, self.poly
        )

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        
        boxes_final = []
        if len(boxes)>0:
            boxes = boxes.astype(np.int32).tolist()
            for box in boxes:
                boxes_final.append([box[0], box[2]])

        return boxes_final
    
    def get_boxes(self, image):
        x, ratio_w, ratio_h = preprocess_image(image, self.canvas_size, self.mag_ratio)
        return self._get_boxes_preprocessed(x, ratio_w, ratio_h)