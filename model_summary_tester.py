import torch
from torchvision import models
from torchsummary import summary
from src.modules.gun_yolo import Gun_Optimized,CustomDarknet53
from holocron.models import darknet53

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Gun_Optimized()
model = model.to(device)
print(summary(model, (1, 1024)))

darknet_model = darknet53(pretrained=True)
model = CustomDarknet53(darknet_model)
model = model.to(device)
print(summary(model, (3, 224,224)))