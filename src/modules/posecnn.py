import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define the CNN model without the dense layer
class poseCNN(nn.Module):
    def __init__(self):
        super(poseCNN, self).__init__()
        
        # Conv2d_1
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # MaxPooling2d_1
        self.max_pooling2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2d_2
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # MaxPooling2d_2
        self.max_pooling2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2d_3
        self.conv2d_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # GlobalAveragePooling2d_1
        self.global_average_pooling2d_1 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = self.max_pooling2d_1(x)
        x = F.relu(self.conv2d_2(x))
        x = self.max_pooling2d_2(x)
        x = F.relu(self.conv2d_3(x))
        fmap = x  # Save the feature map from conv2d_3
        x = self.global_average_pooling2d_1(x)
        gap = x  # Save the feature map from global_average_pooling2d_1
        
        return fmap, gap
