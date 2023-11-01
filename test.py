import numpy as np
from src.modules import motion_analysis, motion_preprocess
import torch
import torch.nn as nn
from torchsummary import summary
from src import model
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from src.modules.posecnn import poseCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

cnn = poseCNN()
print(summary(cnn.cuda(), (1,512,512)) )