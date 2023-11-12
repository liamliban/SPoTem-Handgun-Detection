import cv2
from src.modules import handregion
import torch
import numpy as np
import random

# Set a random seed for reproducibility
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)

torch.backends.cudnn.deterministic=True

# return keypoints dictionary (person_id,keypoints) for one person and plot keypoints
def extract_keypoints(person_id, candidate, subset, confidence_min=0):
    keypoints = {
            'person_id': person_id,
            'keypoints': []
        }
    for kp_id in range(18):  # 18 keypoints for body
        index = int(subset[person_id][kp_id])
        if (not(index == -1)) and candidate[index, 2] >= confidence_min: #if keypoint is detected and confidence score is good, store coordiantes and confidence score
            x = int(candidate[index, 0])
            y = int(candidate[index, 1])
            confidence = candidate[index, 2]
            keypoints['keypoints'].append({
                'x': x,
                'y': y,
                'confidence': confidence
            })
            # Draw keypoints on the image
            # cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)  # Red circles for keypoints
        else: #else, put None coordinates and negative confidence score
            keypoints['keypoints'].append({
                'x': None,
                'y': None,
                'confidence': -1
            })
    return keypoints

# plot body keypoints of one person
def plot_keypoints(canvas,keypoints):
    for kp_id in range(18):
        if keypoints['keypoints'][kp_id]['confidence'] > 0:
            x = keypoints['keypoints'][kp_id]['x']
            y = keypoints['keypoints'][kp_id]['y']
            # Draw keypoints on the image
            cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)  # Red circles for keypoints