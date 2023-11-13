
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

# Define a forward hook to capture the activation of the conv_81 layer
def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class CustomYolo(nn.Module):
    def __init__(self, yolo_model):
        super(CustomYolo, self).__init__()
        self.yolo = yolo_model

        # freeze all parameters of the model
        for param in self.yolo.parameters():
            param.requires_grad = False

        self.target_layer_name = 'conv_81'
        self.target_layer_index = 81
                
        # Initialize the final linear layer
        self.final_linear = nn.Linear(255 * 13 * 13, 20)  # Modify the input size (512) based on your target layer
        
    def forward(self, x):
        # Register a hook to capture the activation of the target layer
        activation = {}
        hook_fn = get_activation(self.target_layer_name, activation)
        target_layer = self.yolo.module_list[self.target_layer_index]
        hook_handle = target_layer.register_forward_hook(hook_fn)
        
        # Pass the input through the pretrained model
        x = self.yolo(x)
        
        # Extract the target activation (e.g., 'conv_81')
        target_activation = activation[self.target_layer_name] #torch.Size([1, 255, 13, 16])
        
        # Remove the hook to avoid memory leaks
        hook_handle.remove()
        
        # Flatten the target activation (modify the size if necessary)
        target_activation_flattened = target_activation.view(target_activation.size(0), -1)
        
        # Pass the target activation through the final linear layer
        x = self.final_linear(target_activation_flattened)

        dense = x
        
        return dense
    

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
    
# class CustomDarknet53_NoDense(nn.Module):
#     def __init__(self, darknet_model):
#         super(CustomDarknet53_NoDense, self).__init__()

#         # Load the pretrained darknet53 model
#         self.darknet = darknet_model

#         # Remove the last linear layer
#         self.darknet2 = torch.nn.Sequential(*list(self.darknet.children())[:-1])

#         # freeze all parameters of the model
#         for param in self.darknet2.parameters():
#             param.requires_grad = False


#     def forward(self, x):
#         x = self.darknet2(x)
#         return x

# class GunLSTM_Optimized(nn.Module): #instead of running darknet53 inside, take the tensors from previously evaluated hand images instead
#     def __init__(self):
#         super(GunLSTM_Optimized, self).__init__()

#         self.lstm = nn.LSTM(
#             input_size=1024, 
#             hidden_size=20, 
#             num_layers=1,
#             batch_first=True)
        
#         # self.dropout = nn.Dropout(0.2)
        
#         # self.linear = nn.Linear(20,20)

#     def forward(self, x):
        
#         r_out, (h_n, h_c) = self.lstm(x)
#         r_out = r_out[:, -1, :]
#         # r_out = self.dropout(r_out)
#         # r_out2 = self.linear(r_out)
        
#         return r_out

# class GunLSTM_Optimized(nn.Module): #instead of running darknet53 inside, take the tensors from previously evaluated hand images instead
#     def __init__(self):
#         super(GunLSTM_Optimized, self).__init__()
#         self.num_layers = 1
#         self.hidden_size = 20
#         self.lstm = nn.LSTM(
#             input_size=1024, 
#             hidden_size=self.hidden_size, 
#             num_layers=self.num_layers,
#             batch_first=True)
#         # self.fc = nn.Linear(self.hidden_size, self.output_size)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

#         out, _ = self.lstm(x, (h0, c0))
        
#         # Select the last time step's output
#         out = out[:, -1, :]
#         # out = self.fc(out)
#         return out
