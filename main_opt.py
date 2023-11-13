import torch
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomYolo, CustomDarknet53, GunLSTM, GunLSTM_Optimized
from src.modules.combined_model import CombinedModel, CombinedModelNewVer
from src.modules.combined_model_no_motion import CombinedModelNoMotion
from src.modules.custom_dataset import CustomGunDataset
from src.modules.custom_dataset_gunLSTM import CustomGunLSTMDataset
from src.modules.custom_dataset_gunLSTM_opt import CustomGunLSTMDataset_opt
from src.modules.train import train_model
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from holocron.models import darknet53

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)


# Load the models
darknet_model = darknet53(pretrained=True)

pose_model = poseCNN()

motion_model = motion_analysis.MotionLSTM()


user_input =  0
while True:
    user_input = input("Do you want to train all three models (1), without motion (2), or the new motion model (3)? Enter '1', '2', or '3': ").strip().upper()
    if user_input == '1':
        gun_model = CustomDarknet53(darknet_model)
        combined_feature_size = 20 + 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModel(gun_model, pose_model, motion_model, combined_feature_size)
        break
    elif user_input == '2':
        gun_model = CustomDarknet53(darknet_model)
        combined_feature_size = 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModelNoMotion(gun_model, pose_model, combined_feature_size)
        break
    elif user_input == '3':
        gun_model = GunLSTM_Optimized()
        combined_feature_size = 20 + 20 #total num of features of 3 model outputs
        combined_model = CombinedModelNewVer(gun_model, pose_model, combined_feature_size)
        break
    else:
        print("Invalid input. Please enter '1' for all three models or '2' for combined model with no motion or '3' for new motion model.")


combined_model.to(device)
combined_model.eval()

window_size = 3

if user_input == '3': 
    # DATASET FOR NEW MODEL
    custom_dataset = CustomGunLSTMDataset_opt(root_dir='data', window_size = window_size)
else:    
    custom_dataset = CustomGunDataset(root_dir='data', window_size = window_size)

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
num_epochs = 20

excel_filename = 'logs/results.xlsx'

# Print Hyperparameters
print(f'''
    Train Set Size: {len(train_dataset)}
    Val Set Size  : {len(val_dataset)}
    Batch Size    : {batch_size}
    Criterion     : {criterion.__class__.__name__}
    Optimizer     : {optimizer.__class__.__name__}
    Learning Rate : {optimizer.param_groups[0]['lr']}
    Epochs        : {num_epochs}
''')

train_losses, val_losses = train_model(user_input, train_loader, val_loader, combined_model, criterion, optimizer, device, num_epochs, excel_filename)

# Add the visualization code here
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

