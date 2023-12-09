import json
import os
import torch
import numpy as np
import random

# Set a random seed for reproducibility
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)
os.environ['PYTHONHASHSEED'] = str(12)
torch.cuda.manual_seed_all(12)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

torch.backends.cudnn.deterministic=True

class PersonIDNotFoundError(Exception):
    pass

# Create file keypoints_seq_[person_id].txt which stores keypoints of frame per line with format
# x0, y0, x1, y1 .... x17, y17
def preprocess_data(keypoints, person_id, folder_path):  
    null_value = 999
    keypoints_sequence = _get_normalized(keypoints, person_id, null_value)

    file_path = _save_keypoints(keypoints_sequence, person_id, folder_path)
    return file_path

# gets the normalized keypoints and return a list (per frame) of a list of body keypoints (x0, y0, x1, y1 .... x17, y17) 
def _get_normalized(normalized_keypoints, person_id, null_value_kps = 999, null_value_person = 0):
    keypoint_sequences = []

    for frame_data in normalized_keypoints: 
        # print(frame_data)
        if len(frame_data) < person_id+1:
            keypoint_set = [null_value_person] * 36
        else:
            person_keypoints = frame_data[person_id]
            if person_keypoints is None:
                keypoint_set = [null_value_person] * 36
            else:
                keypoint_set = []
                for keypoint in person_keypoints:
                    x = keypoint.get("x")
                    y = keypoint.get("y")
                    if x is None:
                        x = null_value_kps

                    if y is None:
                        y = null_value_kps

                    keypoint_set.extend([x, y])
        keypoint_sequences.append(keypoint_set)
    return keypoint_sequences

# takes a keypoints_sequence and save it into a text file
def _save_keypoints(keypoints_sequence, person_id, folder_path):
    # File Path
    file_path = f'{folder_path}keypoints_seq.txt'

    # Check Directory
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(file_path, "w") as text_file:
        for i in range(len(keypoints_sequence)):
            for j in range(36):
                text_file.write('{}'.format(keypoints_sequence[i][j]))
                if j < 35:
                    text_file.write(',')
            text_file.write('\n')

    # Print Log
    print(f'Keypoints sequence stored in: {file_path}')
    return file_path
