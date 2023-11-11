import torch
import torch.nn as nn
from yolo.pytorchyolo import models
from src import model

class CombinedModel(nn.Module):
    def __init__(self, gun_model, pose_model, motion_model, combined_feature_size):
        super(CombinedModel, self).__init__()
        
        self.gun_model = gun_model
        self.pose_model = pose_model
        self.motion_model = motion_model
        self.dense = nn.Linear(combined_feature_size, 2) 

    def forward(self, gun_input, pose_input, motion_input):
        output_gun = self.gun_model(gun_input)
        output_pose = self.pose_model(pose_input)
        output_motion = self.motion_model(motion_input)
        
        # Concatenate the outputs along the feature dimension (dim=1)
        concatenated_output = torch.cat((output_gun, output_pose, output_motion), dim=1)
        
        # Pass the concatenated output through the dense layer
        final_output = self.dense(concatenated_output)
        
        return final_output

class CombinedModelNewVer(nn.Module):
    def __init__(self, gun_model, pose_model, combined_feature_size):
        super(CombinedModelNewVer, self).__init__()
        
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