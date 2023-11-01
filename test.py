import cv2
import os
import torch
import shutil
import src.modules.data_creator as data_creator
from src import model
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomYolo
from src.modules.combined_model import CombinedModel
from src.modules.combined_model_no_motion import CombinedModelNoMotion

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)

# Choose dataset
dataset_folder = 'images/dataset/'
video_label = "8"

# Folder where data are stored
#   -gun: data/[video_label]/hand_image/[person_id]/
#   -pose: data/[video_label]/binary_pose/[person_id]/
#   -motion: data/[video_label]/motion_keypoints/[person_id]/
data_folder = f'./data/'

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
output_folder = data_folder + video_label + "/"

# Clear folder first
if os.path.exists(output_folder):
    for filename in os.listdir(output_folder):
        if os.path.exists(os.path.join(output_folder, filename)):
            shutil.rmtree(output_folder, filename)

num_frames, num_person = data_creator.create_data(dataset_folder, video_label, data_folder, display_animation)