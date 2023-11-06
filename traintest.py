import torch
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomYolo, CustomDarknet53
from src.modules.combined_model import CombinedModel
from src.modules.combined_model_no_motion import CombinedModelNoMotion
from src.modules.custom_dataset import CustomGunDataset
from src.modules.train import train_model
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from holocron.models import darknet53

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)

custom_dataset = CustomGunDataset(root_dir='data')

print ("Number of samples in dataset: ", len(custom_dataset))

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



# call the models
# yolo_model = models.load_model("yolo/config/yolov3.cfg", "yolo/weights/yolov3.weights")
# gun_model = CustomYolo(yolo_model)
darknet_model = darknet53(pretrained=True)
gun_model = CustomDarknet53(darknet_model)

pose_model = poseCNN()

motion_model = motion_analysis.MotionLSTM()



# combined model
#combined_feature_size = 20 + 20 + 20 #total num of features of 3 model outputs
#combined_model = CombinedModel(gun_model, pose_model, motion_model, combined_feature_size)
user_input =  0
while True:
    user_input = input("Do you want to train all three models (1) or without motion (2)? Enter '1' or '2': ").strip().upper()
    if user_input == '1':
        combined_feature_size = 20 + 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModel(gun_model, pose_model, motion_model, combined_feature_size)
        break
    elif user_input == '2':
        combined_feature_size = 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModelNoMotion(gun_model, pose_model, combined_feature_size)
        break
    else:
        print("Invalid input. Please enter '1' for all three models or '2' for combined model with no motion.")


combined_model.to(device)
combined_model.eval()


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
num_epochs = 10

train_losses, val_losses = train_model(user_input, train_loader, val_loader, combined_model, criterion, optimizer, device, num_epochs)

# Add the visualization code here
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

