import torch
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomYolo
from src.modules.combined_model import GPM1
from src.modules.combined_model_no_motion import GP
from src.modules.custom_dataset import CustomGunDataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)

# Specifiy the video. = None if all videos
# video = None
video = '8'

custom_dataset = CustomGunDataset(root_dir='data', video=video)

# print("Dataset number of samples: ", len(custom_dataset))
# for idx, data_entry in enumerate(custom_dataset.data):
#     print(f"Data Entry {idx + 1}:")
#     print("Data Name:", data_entry["data_name"])
#     print("Gun Data Shape:", data_entry["gun_data"].shape)
#     print("Pose Data Shape:", data_entry["pose_data"].shape)
#     print("Label:", data_entry["label"])
#     print("Motion Keypoints:") 
#     print(data_entry["motion_data"])

print ("Number of samples in dataset: ", len(custom_dataset))



# Test one sample in dataset to models

index = 1
data_name, gun_data, pose_data, motion_data, label = custom_dataset[index]
gun_model_input = gun_data
pose_model_input = pose_data
motion_model_input = motion_data


# Print or not print features of models
print_gun_feature = False
print_pose_feature = False
print_motion_feature = False

# GUN MODEL
print("GUN MODEL")
# Load the YOLO model
yolo_model = models.load_model("yolo/config/yolov3.cfg", "yolo/weights/yolov3.weights")

print("\t\tInput shape: " , gun_model_input.shape)

gun_model = CustomYolo(yolo_model)
gun_model.to(device)
gun_model.eval()

if torch.cuda.is_available():
    gun_model_input = gun_model_input.cuda()

gun_feature = gun_model(gun_model_input)
yolo_output = gun_feature

print("\t\tGun Feature Extracted!")
if print_gun_feature:
    print("\t\t", yolo_output)

print("\t\tOutput Shape: ", yolo_output.shape)



# POSE MODEL
print("POSE MODEL")
pose_model = poseCNN()
print("\t\tInput shape: " , pose_model_input.shape)

if torch.cuda.is_available():
    pose_model_input = pose_model_input.cuda()
pose_model.to(device)

pose_feature = pose_model(pose_model_input)

print("\t\tPose Feature Extracted!")
if print_pose_feature:
    print(f"\t\tFeature Map: {pose_feature}")
print("\t\tOutput shape: ", pose_feature.shape)



# MOTION MODEL
print("MOTION MODEL")
if torch.cuda.is_available():
    motion_model_input = motion_model_input.cuda()
print("\t\tInput shape: " , motion_model_input.shape)

motion_model = motion_analysis.MotionLSTM()
motion_model.to(device)

motion_model.eval()  # Set the model in evaluation mode

with torch.no_grad():
    motion_feature = motion_model(motion_model_input)

print("\t\tMotion Feature Extracted!")
if print_motion_feature:
    print("\t\t" , motion_feature)
print("\t\tOutput shape: ", motion_feature.shape)


# COMBINED MODEL
print("COMBINATION MODEL")
combined_feature_size = 20 + 20 + 20 #total num of features of 3 model outputs

combined_model = GPM1(gun_model, pose_model, motion_model, combined_feature_size)
combined_model.to(device)
combined_model.eval()

with torch.no_grad():
    combined_output = combined_model(gun_model_input, pose_model_input, motion_model_input)

print("\t\tCombined Model with Motion Output: ", combined_output)

# COMBINED MODEL NO MOTION
print("COMBINATION MODEL no Motion")
combined_2_feature_size = 20 + 20 #total num of features of 2 model outputs

combined_model_2 = GP(gun_model, pose_model, combined_2_feature_size)
combined_model_2.to(device)
combined_model_2.eval()

with torch.no_grad():
    combined_output_2 = combined_model_2(gun_model_input, pose_model_input)

print("\t\tCombined Model without Motion Output: ", combined_output_2)

