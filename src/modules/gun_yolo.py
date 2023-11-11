
import torch
import torch.nn as nn
from src import model
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from torch.nn import MaxPool2d, functional as F
from src.modules.utils import GlobalAvgPool2d, auto_pad

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


class CustomDarknet53_Standalone(nn.Module):
    def __init__(self, darknet_model):
        super(CustomDarknet53_Standalone, self).__init__()

        # Load the pretrained darknet53 model
        self.darknet = darknet_model

        # Remove the last linear layer
        self.darknet2 = torch.nn.Sequential(*list(self.darknet.children())[:-1])

        # freeze all parameters of the model
        for param in self.darknet2.parameters():
            param.requires_grad = False

        # Add a final linear layer
        self.dense = nn.Linear(1024, 2)

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
        
        # self.linear = nn.Linear(20,20)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        r_out = r_out[:, -1, :]
        # r_out2 = self.linear(r_out)
        
        return r_out



# class Conv(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, auto_pad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.LeakyReLU(0.01) if act else nn.ReLU()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))


# class ResidualBlock(nn.Module):
#     def __init__(self, c1, shortcut=True):
#         super(ResidualBlock, self).__init__()
#         c2 = c1 // 2
#         self.shortcut = shortcut
#         self.layer1 = Conv(c1, c2, p=0)
#         self.layer2 = Conv(c2, c1, k=3)

#     def forward(self, x):
#         residual = x
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out += residual
#         if self.shortcut:
#             out += residual
#         return out
    
# class DarkNet53FeatureExtractor(nn.Module):
#     def __init__(self, block):
#         super(DarkNet53FeatureExtractor, self).__init__()

#         self.features = nn.Sequential(
#             Conv(3, 32, 3),
#             Conv(32, 64, 3, 2),
#             *self._make_layer(block, 64, num_blocks=1),
#             Conv(64, 128, 3, 2),
#             *self._make_layer(block, 128, num_blocks=2),
#             Conv(128, 256, 3, 2),
#             *self._make_layer(block, 256, num_blocks=8),
#             Conv(256, 512, 3, 2),
#             *self._make_layer(block, 512, num_blocks=8),
#             Conv(512, 1024, 3, 2),
#             *self._make_layer(block, 1024, num_blocks=4),
#             GlobalAvgPool2d()
#         )

#         self.final_linear = nn.Linear(1024, 20)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.final_linear(x)
#         return x

#     @staticmethod
#     def _make_layer(block, in_channels, num_blocks):
#         layers = []
#         for i in range(0, num_blocks):
#             layers.append(block(in_channels))
#         return nn.Sequential(*layers)

# class DarkNet53(nn.Module):
#     """ [https://pjreddie.com/media/files/papers/YOLOv3.pdf] """
#     def __init__(self, block, num_classes=1000, init_weight=True):
#         super(DarkNet53, self).__init__()
#         self.num_classes = num_classes

#         if init_weight:
#             self._initialize_weights()

#         self.features = nn.Sequential(
#             Conv(3, 32, 3),

#             Conv(32, 64, 3, 2),
#             *self._make_layer(block, 64, num_blocks=1),

#             Conv(64, 128, 3, 2),
#             *self._make_layer(block, 128, num_blocks=2),

#             Conv(128, 256, 3, 2),
#             *self._make_layer(block, 256, num_blocks=8),

#             Conv(256, 512, 3, 2),
#             *self._make_layer(block, 512, num_blocks=8),

#             Conv(512, 1024, 3, 2),
#             *self._make_layer(block, 1024, num_blocks=4)
#         )
#         self.classifier = nn.Sequential(
#             *self.features,
#             GlobalAvgPool2d(),
#             nn.Linear(1024, num_classes)
#         )

#     def forward(self, x):
#         return self.classifier(x)

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     @staticmethod
#     def _make_layer(block, in_channels, num_blocks):
#         layers = []
#         for i in range(0, num_blocks):
#             layers.append(block(in_channels))
#         return nn.Sequential(*layers)