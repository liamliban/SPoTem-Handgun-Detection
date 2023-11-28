import torch
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomYolo, CustomDarknet53, GunLSTM
from src.modules.combined_model import CombinedModel
from src.modules.combined_model_no_motion import CombinedModelNoMotion
from src.modules.custom_dataset_gunLSTM import CustomGunLSTMDataset
from src.modules.train_gunlstm import train_model
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from holocron.models import darknet53

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)

# Specifiy the video. = None if all videos
# video = '8'
# custom_dataset = CustomGunDataset(root_dir='data', video=video)

# create dataset for all generated data
custom_dataset = CustomGunLSTMDataset(root_dir='data')

print("Dataset number of samples: ", len(custom_dataset))
for idx, data_entry in enumerate(custom_dataset.data):
    print(f"Data Entry {idx + 1}:")
    print("Data Name:", data_entry["data_name"])
    print("Gun Data Shape:", data_entry["gun_data"].shape)
    print("Pose Data Shape:", data_entry["pose_data"].shape)
    print("Label:", data_entry["label"])

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
    
# Test one sample in dataset to models
index = 3

# get the data from dataset sample
data_name, gun_data, pose_data, label = custom_dataset[index]

pose_model_input = pose_data.unsqueeze(0)

gun_model_input = gun_data.unsqueeze(0)

if torch.cuda.is_available():
    gun_model_input = gun_model_input.cuda()
    pose_model_input = pose_model_input.cuda()

print("gun input size: ", gun_model_input.shape)


# call the models
# yolo_model = models.load_model("yolo/config/yolov3.cfg", "yolo/weights/yolov3.weights")
# gun_model = CustomYolo(yolo_model)
darknet_model = darknet53(pretrained=True)
gun_model = GunLSTM(darknet_model)

pose_model = poseCNN()

# TEST GUNLSTM
gun_model.to(device)
gun_model.eval()

with torch.no_grad():
    gun_output = gun_model(gun_model_input)

print("Gun Model ouput: ", gun_output)
print("Gun Model output shape: ", gun_output.shape)




# combined model
combined_feature_size = 20 + 20#total num of features of 3 model outputs
combined_model = CombinedModelNoMotion(gun_model, pose_model, combined_feature_size)


combined_model.to(device)
combined_model.eval()

with torch.no_grad():
    combined_output = combined_model(gun_model_input, pose_model_input)

print("Combined Model with Motion Output: ", combined_output)








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

train_losses, val_losses = train_model(2, train_loader, val_loader, combined_model, criterion, optimizer, device, num_epochs, excel_filename)

# Add the visualization code here
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
