import os
import torch
import shutil
import re
import src.modules.data_creator as data_creator
from src.modules import annotator
import numpy as np
import random
import os

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

def create_annotate_all_videos(dataset_folder, data_folder):
    video_names_list = _list_subfolders(dataset_folder)

    print("All videos : ", video_names_list)

    for video_name in video_names_list:
        print("Generating and Annotating data of Video: ", video_name)
        _create_annotate_video(dataset_folder, data_folder, video_name)
        print("Data generated and annotated fro video: ", video_name)

# Get all video names
def _list_subfolders(main_folder_path):
    subfolders = []

    # Custom sorting key function
    def custom_sort_key(item):
        # Split the string into parts using underscores
        parts = item.split('_')
        if len(parts) == 3:
            # For numeric strings, return a tuple (0, int) to sort them first
            return (1, int(parts[2]))
        else:
            # For alphanumeric strings, return a tuple (1, int) to sort them after the numeric ones
            return (0, int(parts[-1]))
    
    # Check if the main folder path exists
    if os.path.exists(main_folder_path) and os.path.isdir(main_folder_path):
        for folder_name in os.listdir(main_folder_path):
            folder_path = os.path.join(main_folder_path, folder_name)
            if os.path.isdir(folder_path):
                subfolders.append(folder_name)
    
    return sorted(subfolders, key=custom_sort_key)



def _create_annotate_video(dataset_folder, data_folder, video_name):
    # File names of data:
    #   -gun: hands_[frame_num].png
    #   -pose: pose_[frame_num].png
    #   -motion: keypoints_seq.txt

    # create:
    #   -hand region images (gun), 
    #   -binary pose image (pose), 
    #   -preprocessed keypoints text file (motion)
    display_animation = False
    # Path of output video folder
    output_folder = data_folder + video_name + "/"

    # Clear folder first
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            if os.path.exists(os.path.join(output_folder, filename)):
                shutil.rmtree(output_folder, filename)

    data_creator.create_data(dataset_folder, video_name, data_folder, display_animation)

    # folder where the annotations are stored
    annotation_folder = "raw_dataset/annotations/"

    # Create video annotation
    video_labels = annotator.create_vid_annotation(dataset_folder, data_folder, video_name, output_folder, annotation_folder)

    if video_labels is not None:
        # Save video annotation
        annotator.save_video_labels_csv(video_labels, output_folder)




# Get folder where raw dataset is stored
dataset_folder = 'raw_dataset/dataset/'

# Folder where data are stored
#   -gun: data/[video_label]/[person_id]/hand_image/
#   -pose: data/[video_label]/[person_id]/binary_pose/
#   -motion: data/[video_label]/[person_id]/motion_keypoints/
data_folder = f'./data/'

create_annotate_all_videos(dataset_folder, data_folder)






