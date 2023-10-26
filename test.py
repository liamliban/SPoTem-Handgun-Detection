import cv2
import os
import json
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src import model
from src.body import Body
from src import util
from src.modules import handregion, bodykeypoints, handimage
from src.modules.binarypose import BinaryPose
from yolo.pytorchyolo import detect, models
import torchvision.transforms as transforms

# Initialize body estimation model
body_estimation = Body('model/body_pose_model.pth')

# Load the YOLO model
model = models.load_model(
  "yolov3.cfg", "yolov3.weights")


# Load the image as a numpy array
img = cv2.imread("images/dataset/3/CA00072.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#print(img.shape)
# img = np.einsum('ijk->kij', img)
print(img.shape)
img = transforms.ToTensor()(img)
#img = img.permute(1, 2, 0)
print(img.size())
img = img.unsqueeze(0)
print(img.size()) 
print(img)

#activation = {}
#def get_activation(name):
#    def hook(model, input, output):
#        activation[name] = output.detach()
#    return hook
#model.fc3.register_forward_hook(get_activation('fc3'))
#output = model(x)
#activation['fc3']

#print(model.forward)
model.eval()
print(model(img))