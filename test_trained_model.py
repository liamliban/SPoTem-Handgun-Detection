import torch
from src.modules import motion_analysis
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomDarknet53, GunLSTM, GunLSTM_Optimized, Gun_Optimized
from src.modules.combined_model import CombinedModel, CombinedModelNewVer
from src.modules.combined_model_no_motion import CombinedModelNoMotion
from src.modules.custom_dataset import CustomGunDataset
from src.modules.custom_dataset_gunLSTM import CustomGunLSTMDataset
from src.modules.custom_dataset_gunLSTM_opt import CustomGunLSTMDataset_opt
from src.modules.custom_dataset_opt import CustomGunDataset_opt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from holocron.models import darknet53

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)

# Change Checkpoint File Location here
checkpoint_path = 'E:/Thesis/Thesis-Gun-Detection/logs/run#53/fold1/model/model_epoch_1.pt'
print('/'.join(checkpoint_path.split("/")[:-2]))

checkpoint = torch.load(checkpoint_path) 

hidden_size = checkpoint['model_info']['hidden_size']
window_size = checkpoint['model_info']['window_size']
lstm_layers = checkpoint['model_info']['lstm_layers']
model_type = checkpoint['model_info']['model_type']

# Load Model and Data Based on Model Type
darknet_model = darknet53(pretrained=True)
pose_model = poseCNN()
if model_type == 'GPM':
    gun_model = CustomDarknet53(darknet_model)
    motion_model = motion_analysis.MotionLSTM(hidden_size, lstm_layers)
    combined_feature_size = 20 + 20 + hidden_size #total num of features of 3 model outputs
    trained_model = CombinedModel(gun_model, pose_model, motion_model, combined_feature_size)
    custom_dataset = CustomGunDataset(root_dir='data', window_size = window_size)
elif model_type == 'GPM2':
    gun_model = GunLSTM(darknet_model, hidden_size=hidden_size)
    combined_feature_size = 20 + hidden_size #total num of features of 3 model outputs
    trained_model = CombinedModelNewVer(gun_model, pose_model, combined_feature_size)
    custom_dataset = CustomGunLSTMDataset(root_dir='data', window_size = window_size)
elif model_type == 'GP':
    gun_model = CustomDarknet53(darknet_model)
    combined_feature_size = 20 + 20 #total num of features of 3 model outputs
    trained_model = CombinedModelNoMotion(gun_model, pose_model, combined_feature_size)
    custom_dataset = CustomGunDataset(root_dir='data', window_size = window_size)
elif model_type == 'GPM-opt':
    gun_model = Gun_Optimized()
    motion_model = motion_analysis.MotionLSTM(hidden_size, lstm_layers)
    combined_feature_size = 20 + 20 + hidden_size #total num of features of 3 model outputs
    trained_model = CombinedModel(gun_model, pose_model, motion_model, combined_feature_size)
    custom_dataset = CustomGunDataset_opt(root_dir='data', window_size = window_size)
elif model_type == 'GPM2-opt':
    gun_model = GunLSTM_Optimized(hidden_size, lstm_layers)
    combined_feature_size = 20 + hidden_size #total num of features of 3 model outputs
    trained_model = CombinedModelNewVer(gun_model, pose_model, combined_feature_size)
    custom_dataset = CustomGunLSTMDataset_opt(root_dir='data', window_size = window_size)
else:
    gun_model = Gun_Optimized()
    combined_feature_size = 20 + 20 #total num of features of 3 model outputs
    trained_model = CombinedModelNewVer(gun_model, pose_model, combined_feature_size)
    custom_dataset = CustomGunDataset_opt(root_dir='data', window_size = window_size)

trained_model.load_state_dict(checkpoint['model_state_dict'])
trained_model.to(device)
_, val_dataset = train_test_split(custom_dataset, test_size=0.2, random_state=42)

# Look into correct and incorrect predictions
incorrect_predictions = []
video_loader = DataLoader(val_dataset)
trained_model.eval()
with torch.no_grad():
    for index, video in enumerate(video_loader):
        print(f'Predicting...[{index}/{len(video_loader)}]', end='\r')
        data_name, gun_data, pose_data, motion_data, target_label = video

        gun_data = gun_data.to(device)
        pose_data = pose_data.to(device)
        motion_data = motion_data.to(device)
        target_label = target_label.to(device)
        
        if model_type == 'GPM' or model_type == 'GPM-opt':
            combined_output = trained_model(gun_data, pose_data, motion_data)
        else:
            combined_output = trained_model(gun_data, pose_data)
        
        predicted_label = torch.argmax(combined_output)
        if predicted_label != target_label:
            incorrect_predictions.append({'data_name': data_name[0], 'predicted_label': predicted_label.item(), 'target_label': target_label.item()})
    print(f'Predicting...[{len(video_loader)}/{len(video_loader)}]')

# Show and Save Incorrect Predictions
run_name = checkpoint_path.split("/")[1]
incorrect_predictions_path = '/'.join(checkpoint_path.split("/")[:-2]) + '/IncorrectPredictions.txt'
with open(incorrect_predictions_path, 'w') as file:
    file.write('Incorrect Predictions:\n\n')
    for item in incorrect_predictions:
        print(item['data_name'] + ':')
        print(f'  Predicted Label: {item["predicted_label"]}')
        print(f'  Correct Label: {item["target_label"]}')
        print()
        file.write(item['data_name'] + ':\n')
        file.write(f'  Predicted Label: {item["predicted_label"]}\n')
        file.write(f'  Correct Label: {item["target_label"]}\n\n')
    print(f'Incorrect Predictions saved to: {incorrect_predictions_path}')