import torch
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomYolo
from src.modules.combined_model import CombinedModel
from src.modules.combined_model_no_motion import CombinedModelNoMotion
from src.modules.custom_dataset import CustomGunDataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)

# Specifiy the video. = None if all videos
# video = '8'
# custom_dataset = CustomGunDataset(root_dir='data', video=video)

# create dataset for all generated data
custom_dataset = CustomGunDataset(root_dir='data')

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

# get the 3 data from dataset sample
data_name, gun_data, pose_data, motion_data, label = custom_dataset[index]
gun_model_input = gun_data
pose_model_input = pose_data
motion_model_input = motion_data

if torch.cuda.is_available():
    gun_model_input = gun_model_input.cuda()
    pose_model_input = pose_model_input.cuda()
    motion_model_input = motion_model_input.cuda()


# call the models
yolo_model = models.load_model("yolo/config/yolov3.cfg", "yolo/weights/yolov3.weights")
gun_model = CustomYolo(yolo_model)

pose_model = poseCNN()

motion_model = motion_analysis.MotionLSTM()

# combined model
combined_feature_size = 20 + 20 + 20 #total num of features of 3 model outputs
combined_model = CombinedModel(gun_model, pose_model, motion_model, combined_feature_size)


combined_model.to(device)
combined_model.eval()

with torch.no_grad():
    combined_output = combined_model(gun_model_input, pose_model_input, motion_model_input)

print("Combined Model with Motion Output: ", combined_output)

