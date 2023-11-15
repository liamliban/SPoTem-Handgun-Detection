import torch
import os
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomDarknet53, GunLSTM, GunLSTM_Optimized, Gun_Optimized
from src.modules.combined_model import CombinedModel, CombinedModelNewVer
from src.modules.combined_model_no_motion import CombinedModelNoMotion
from src.modules.custom_dataset import CustomGunDataset
from src.modules.custom_dataset_gunLSTM import CustomGunLSTMDataset
from src.modules.custom_dataset_gunLSTM_opt import CustomGunLSTMDataset_opt
from src.modules.custom_dataset_opt import CustomGunDataset_opt
from src.modules.train import train_model
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from holocron.models import darknet53
import numpy as np
import random

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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)


# Load the models
darknet_model = darknet53(pretrained=True)

pose_model = poseCNN()

motion_model = motion_analysis.MotionLSTM()


user_input =  0
model_name = ''
while True:
    user_input = input("Do you want to train GPM (1), GP (2), GPM2 (3), GPM2-opt (4), GP-opt (5), GPM-opt (6)? Enter '1', '2', '3', '4', '5', '6': ").strip().upper()
    if user_input == '1':
        gun_model = CustomDarknet53(darknet_model)
        combined_feature_size = 20 + 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModel(gun_model, pose_model, motion_model, combined_feature_size)
        model_name = 'GPM'
        break
    elif user_input == '2':
        gun_model = CustomDarknet53(darknet_model)
        combined_feature_size = 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModelNoMotion(gun_model, pose_model, combined_feature_size)
        model_name = 'GP'
        break
    elif user_input == '3':
        gun_model = GunLSTM(darknet_model)
        combined_feature_size = 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModelNewVer(gun_model, pose_model, combined_feature_size)
        model_name = 'GPM2'
        break
    elif user_input == '4':
        gun_model = GunLSTM_Optimized()
        combined_feature_size = 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModelNewVer(gun_model, pose_model, combined_feature_size)
        model_name = 'GPM2-opt'
        break
    elif user_input == '5':
        gun_model = Gun_Optimized()
        combined_feature_size = 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModelNewVer(gun_model, pose_model, combined_feature_size)
        model_name = 'GP-opt'
        break
    elif user_input == '6':
        gun_model = Gun_Optimized()
        combined_feature_size = 20 + 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModel(gun_model, pose_model, motion_model, combined_feature_size)
        model_name = 'GPM-opt'
        break
    else:
        print("Invalid input. Please enter '1' for all three models or '2' for combined model with no motion or '3' for new motion model.")


combined_model.to(device)
combined_model.eval()

window_size = 3

if user_input == '3': 
    # DATASET FOR NEW MODEL
    custom_dataset = CustomGunLSTMDataset(root_dir='data', window_size = window_size)
elif user_input == '4': 
    # DATASET FOR NEW MODEL optimized
    custom_dataset = CustomGunLSTMDataset_opt(root_dir='data', window_size = window_size)
elif user_input == '5' or user_input == '6': 
    # DATASET FOR old model optimized
    custom_dataset = CustomGunDataset_opt(root_dir='data', window_size = window_size)
else:    
    custom_dataset = CustomGunDataset(root_dir='data', window_size = window_size)

print ("Number of samples in dataset: ", len(custom_dataset))

# for idx, data_entry in enumerate(custom_dataset.data):
#     print(f"Data Entry {idx + 1}:")
#     print("Data Name:", data_entry["data_name"])
#     print("Gun Data Shape:", data_entry["gun_data"].shape)
#     print("Pose Data Shape:", data_entry["pose_data"].shape)

label_0 = 0
label_1 = 0

for idx, data_entry in enumerate(custom_dataset.data):
    label = data_entry["label"]
    if label == '0':
        label_0+=1
    elif label == '1':
        label_1+=1

print ("Number of label 0: ", label_0)
print ("Number of label 1: ", label_1)


# Split the dataset into training and validation sets
train_dataset, val_dataset = train_test_split(custom_dataset, test_size=0.2, random_state=42)

batch_size = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

# Set the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Training loop
num_epochs = 60

excel_filename = 'logs/results.xlsx'

# Print Hyperparameters
print(f'''
    Model Type    : {model_name}
    Window Size   : {window_size}
    Train Set Size: {len(train_dataset)}
    Val Set Size  : {len(val_dataset)}
    Batch Size    : {batch_size}
    Criterion     : {criterion.__class__.__name__}
    Optimizer     : {optimizer.__class__.__name__}
    Learning Rate : {optimizer.param_groups[0]['lr']}
    Epochs        : {num_epochs}
''')

train_losses, val_losses = train_model(user_input, train_loader, val_loader, combined_model, criterion, optimizer, device, num_epochs, excel_filename, window_size=window_size)