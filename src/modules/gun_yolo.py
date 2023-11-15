
import torch
import torch.nn as nn
from src import model
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from torch.nn import MaxPool2d, functional as F
from src.modules.utils import GlobalAvgPool2d, auto_pad
import numpy as np
import random, os

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

class CustomDarknet53(nn.Module):
    def __init__(self, darknet_model):
        super(CustomDarknet53, self).__init__()

        # Load the pretrained darknet53 model
        self.darknet = darknet_model

        # Remove the last linear layer
        self.darknet2 = torch.nn.Sequential(*list(self.darknet.children())[:-1])

        # freeze all parameters of the model
        for param in self.darknet2.parameters():
            param.requires_grad = False

        # Add a final linear layer
        self.dense = nn.Linear(1024, 20)

    def forward(self, x):
        x = self.darknet2(x)
        x = self.dense(x)
        return x


class GunLSTM(nn.Module):
    def __init__(self, darknet_model):
        super(GunLSTM, self).__init__()

        # Load the pretrained darknet53 model
        self.darknet = darknet_model

        # Remove the last linear layer
        self.darknet2 = torch.nn.Sequential(*list(self.darknet.children())[:-1])

        # freeze all parameters of the model
        for param in self.darknet2.parameters():
            param.requires_grad = False

        self.cnn = self.darknet2
        self.lstm = nn.LSTM(
            input_size=1024, 
            hidden_size=20, 
            num_layers=1,
            batch_first=True)
        
        # self.dropout = nn.Dropout(0.4)
        
        # self.linear = nn.Linear(20,20)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out = r_out[:, -1, :]

        # r_out = self.dropout(r_out)
        # r_out2 = self.linear(r_out)
        
        return r_out

class CustomDarknet53_NoDense(nn.Module):
    def __init__(self, darknet_model):
        super(CustomDarknet53_NoDense, self).__init__()

        # Load the pretrained darknet53 model
        self.darknet = darknet_model

        # Remove the last linear layer
        self.darknet2 = torch.nn.Sequential(*list(self.darknet.children())[:-1])

        # freeze all parameters of the model
        for param in self.darknet2.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.darknet2(x)
        return x
    
class Gun_Optimized(nn.Module):
    def __init__(self):
        super(Gun_Optimized, self).__init__()

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(1024, 20)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        return x

class GunLSTM_Optimized(nn.Module): #instead of running darknet53 inside, take the tensors from previously evaluated hand images instead
    def __init__(self):
        super(GunLSTM_Optimized, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1024, 
            hidden_size=20, 
            num_layers=1,
            batch_first=True)
        
        # self.dropout = nn.Dropout(0.2)
        
        # self.linear = nn.Linear(20,20)

    def forward(self, x):
        
        r_out, (h_n, h_c) = self.lstm(x)
        r_out = r_out[:, -1, :]
        # r_out = self.dropout(r_out)
        # r_out2 = self.linear(r_out)
        
        return r_out

