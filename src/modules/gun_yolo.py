
import torch
import torch.nn as nn
from src import model
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms

# Define a forward hook to capture the activation of the conv_81 layer
def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class CustomYolo(nn.Module):
    def __init__(self, yolo_model):
        super(CustomYolo, self).__init__()
        self.yolo = yolo_model
        self.target_layer_name = 'conv_81'
        self.target_layer_index = 81
                
        # Initialize the final linear layer
        self.final_linear = nn.Linear(255 * 13 * 16, 20)  # Modify the input size (512) based on your target layer
        
    def forward(self, x):
        # Register a hook to capture the activation of the target layer
        activation = {}
        hook_fn = get_activation(self.target_layer_name, activation)
        target_layer = self.yolo.module_list[self.target_layer_index]
        hook_handle = target_layer.register_forward_hook(hook_fn)
        
        # Pass the input through the pretrained model
        x = self.yolo(x)
        
        # Extract the target activation (e.g., 'conv_81')
        target_activation = activation[self.target_layer_name]
        
        # Remove the hook to avoid memory leaks
        hook_handle.remove()
        
        # Flatten the target activation (modify the size if necessary)
        target_activation_flattened = target_activation.view(target_activation.size(0), -1)
        
        # Pass the target activation through the final linear layer
        x = self.final_linear(target_activation_flattened)

        dense = x
        
        return target_activation, dense