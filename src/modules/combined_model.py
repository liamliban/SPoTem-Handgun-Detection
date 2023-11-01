import torch
import torch.nn as nn
from src import model

class CombinedModel(nn.Module):
    def __init__(self, gun_model, pose_model, motion_model, combined_feature_size):
        super(CombinedModel, self).__init__()
        
        self.gun_model = gun_model
        self.pose_model = pose_model
        self.motion_model = motion_model
        self.dense = nn.Linear(combined_feature_size, 2) 

    def forward(self, x1, x2, x3):
        output_gun = self.gun_model(x1)
        output_pose = self.pose_model(x2)
        output_motion = self.motion_model(x3)
        
        # Concatenate the outputs along the feature dimension (dim=1)
        concatenated_output = torch.cat((output_gun, output_pose, output_motion), dim=1)
        
        # Pass the concatenated output through the dense layer
        final_output = self.dense(concatenated_output)
        
        return final_output