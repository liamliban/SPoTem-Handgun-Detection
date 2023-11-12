import torch
import torch.nn as nn
from src import model
import random
import numpy as np

# Set a random seed for reproducibility
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)

torch.backends.cudnn.deterministic=True

class CombinedModelNoMotion(nn.Module):
    def __init__(self, gun_model, pose_model, combined_feature_size):
        super(CombinedModelNoMotion, self).__init__()
        
        self.gun_model = gun_model
        self.pose_model = pose_model
        self.dense = nn.Linear(combined_feature_size, 2) 

    def forward(self, gun_input, pose_input):
        output_gun = self.gun_model(gun_input)
        output_pose = self.pose_model(pose_input)
        
        # Concatenate the outputs along the feature dimension (dim=1)
        concatenated_output = torch.cat((output_gun, output_pose), dim=1)
        
        # Pass the concatenated output through the dense layer
        final_output = self.dense(concatenated_output)
        
        return final_output