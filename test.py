import numpy as np
from src import model
from src.modules import motion_analysis, motion_preprocess
import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

keypoints_file_path = "normalized_keypoints_data.json"
video_label = "test"
person_id = 0

# get path of text file containing preprocessed data (specified by video and person id)
data_path = motion_preprocess.preprocess_data(keypoints_file_path, video_label, person_id)

window_size = 3
data = motion_analysis.load_data(data_path, window_size)
frame_num = 2 #not less than window_size - 1
if frame_num < window_size - 1:
    print("Motion Analysis: Not enough previous frames. No feature extracted")
else:
  data = data[frame_num - (window_size - 1)].unsqueeze(0) #get one sequence only
  print(data)

  # Define the model and specify hyperparameters
  input_size = 36
  hidden_size = 64
  num_layers = 1
  output_size = 1

  motion_model = motion_analysis.MotionLSTM(input_size, hidden_size, num_layers, output_size)
  motion_model.to(device)

  motion_model.eval()  # Set the model in evaluation mode

  with torch.no_grad():
      outputs = motion_model(data)

  print(outputs)
  print(outputs.shape)