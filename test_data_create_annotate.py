import os
import torch
import shutil
import src.modules.data_creator as data_creator
from src.modules import annotator

# Choose dataset
dataset_folder = 'raw_dataset/dataset/'
video_name = "16"

# Folder where data are stored
#   -gun: data/[video_label]/[person_id]/hand_image/
#   -pose: data/[video_label]/[person_id]/binary_pose/
#   -motion: data/[video_label]/[person_id]/motion_keypoints/
data_folder = f'./data/'

# File names of data:
#   -gun: hands_[frame_num].png
#   -pose: pose_[frame_num].png
#   -motion: keypoints_seq.txt

# create:
#   -hand region images (gun), 
#   -binary pose image (pose), 
#   -preprocessed keypoints text file (motion)
display_animation = True
# Path of output video folder
output_folder = data_folder + video_name + "/"

# Clear folder first
if os.path.exists(output_folder):
    for filename in os.listdir(output_folder):
        if os.path.exists(os.path.join(output_folder, filename)):
            shutil.rmtree(output_folder, filename)

data_creator.create_data(dataset_folder, video_name, data_folder, display_animation)

# folder where the generated data by data_creator is stored
data_folder = "data/"

# folder where the annotations are stored
annotation_folder = "raw_dataset/annotations/"

# Create video annotation
video_labels = annotator.create_vid_annotation(dataset_folder, data_folder, video_name, output_folder, annotation_folder)

if video_labels is not None:
    # Save video annotation
    annotator.save_video_labels_csv(video_labels, output_folder)

    # Get num of frames and persons from the video labels csv file
    num_frames, num_person = data_creator.get_num_frames_person(data_folder, video_name)