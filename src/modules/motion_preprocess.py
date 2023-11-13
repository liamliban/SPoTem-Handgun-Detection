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
def preprocess_data(keypoints_file_path, person_id, folder_path):  
    null_value = 999
    keypoints_sequence = _get_normalized(keypoints_file_path, person_id, null_value)

    file_path = _save_keypoints(keypoints_sequence, person_id, folder_path)
    return file_path

# gets the normalized keypoints and return a list (per frame) of a list of body keypoints (x0, y0, x1, y1 .... x17, y17) 
def _get_normalized(normalized_keypoints, person_id, null_value):
    keypoint_sequences = []

    for frame_data in normalized_keypoints: 
        keypoints = frame_data.get("keypoints") #each frame data has "keypoints" (keypoints of all persons)

        if len(keypoints) == 0: #if no person detected on the frame
            keypoint_sequences.append([null_value] * 36)
        else:
            keypoint_set = []

            person_id_found = False
            for person_data in keypoints: #check all persons in the frame
                #if person id match, get keypoint set
                if person_data.get("person_id") == person_id: 
                    person_id_found = True
                    person_keypoints = person_data.get("keypoints")

                    for keypoint in person_keypoints:
                        x = keypoint.get("x")
                        y = keypoint.get("y")
                        if x is None:
                            x = null_value

                        if y is None:
                            y = null_value

                        keypoint_set.extend([x, y])
                    # print("keypoint set if yes: " , keypoint_set)
                    
                    break #stop looking for person id if already found

            # if person id is not found, add null values
            if not person_id_found:
                keypoint_set = [null_value] * 36

                
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
