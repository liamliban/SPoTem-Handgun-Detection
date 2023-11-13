import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import random

# Set a random seed for reproducibility
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)

torch.backends.cudnn.deterministic=True

# Define the CNN model without the dense layer
class poseCNN(nn.Module):
    def __init__(self):
        super(poseCNN, self).__init__()
        
        # Conv2d_1
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0)
        
        # MaxPooling2d_1
        self.max_pooling2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2d_2
        self.conv2d_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        
        # MaxPooling2d_2
        self.max_pooling2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2d_3
        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        
        # GlobalAveragePooling2d_1
        self.global_average_pooling2d_1 = nn.AdaptiveAvgPool2d(1)

        # Linear layer
        self.dense = nn.Linear(32, 20)

    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = self.max_pooling2d_1(x)
        x = F.relu(self.conv2d_2(x))
        x = self.max_pooling2d_2(x)
        x = F.relu(self.conv2d_3(x))
        fmap = x  # Save the feature map from conv2d_3
        x = self.global_average_pooling2d_1(x)
        gap = x  # Save the feature map from global_average_pooling2d_1
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        dense = x

        return dense
