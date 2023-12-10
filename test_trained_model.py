import torch
from src.modules import motion_analysis
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomDarknet53, GunLSTM, GunLSTM_Optimized, Gun_Optimized
from src.modules.combined_model import GPM1, GPM2
from src.modules.custom_dataset import CustomGunDataset
from src.modules.custom_dataset_gunLSTM import CustomGunLSTMDataset
from src.modules.custom_dataset_gunLSTM_opt import CustomGunLSTMDataset_opt
from src.modules.custom_dataset_opt import CustomGunDataset_opt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from holocron.models import darknet53
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import itertools

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)

def plot_confusion_matrix(matrix, classes, title, path):
    """
    Plot and save the confusion matrix as an image.
    """
    plt.figure(figsize=(len(classes) + 2, len(classes) + 2))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black",
                 fontsize=10)

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    plt.savefig(path)
    print(f'Confusion Matrix Saved To: {path}')
    plt.close()

# VALUES TO CHANGE:
root_folder = f'logs/Cross Validation GPM1'
root_dir = 'data2'

print(f'Root Folder: {root_folder}')
print(f'Data Folder: {root_dir}')

checkpoint_path = f'{root_folder}/fold1/model/model_epoch_59.pt' # use fold 1 to get hyperparams
checkpoint = torch.load(checkpoint_path)
hidden_size = checkpoint['model_info']['hidden_size']
window_size = checkpoint['model_info']['window_size']
lstm_layers = checkpoint['model_info']['lstm_layers']
model_type = checkpoint['model_info']['model_type']
print(f'Model Type: {model_type}')

# LOAD DATA
if model_type == 'GPM':
    custom_dataset = CustomGunDataset(root_dir=root_dir, window_size = window_size)
elif model_type == 'GPM2':
    custom_dataset = CustomGunLSTMDataset(root_dir=root_dir, window_size = window_size)
elif model_type == 'GP':
    custom_dataset = CustomGunDataset(root_dir=root_dir, window_size = window_size)
elif model_type == 'GPM-opt':
    custom_dataset = CustomGunDataset_opt(root_dir=root_dir, window_size = window_size)
elif model_type == 'GPM2-opt':
    custom_dataset = CustomGunLSTMDataset_opt(root_dir=root_dir, window_size = window_size)
else:
    custom_dataset = CustomGunDataset_opt(root_dir=root_dir, window_size = window_size)

folds = 5
kf = KFold(n_splits=folds, random_state=42, shuffle=True)
for fold_num, (_, val_indices) in enumerate(kf.split(custom_dataset)):

    # Reload model per fold
    checkpoint_path = f'{root_folder}/fold{fold_num+1}/model/model_epoch_59.pt' # do not change
    checkpoint = torch.load(checkpoint_path)

    # Load Model and Data Based on Model Type
    darknet_model = darknet53(pretrained=True)
    pose_model = poseCNN()

    # LOAD MODEL
    if model_type == 'GPM':
        gun_model = CustomDarknet53(darknet_model)
        motion_model = motion_analysis.MotionLSTM(hidden_size, lstm_layers)
        combined_feature_size = 20 + 20 + hidden_size #total num of features of 3 model outputs
        trained_model = GPM1(gun_model, pose_model, motion_model, combined_feature_size)
    elif model_type == 'GPM2':
        gun_model = GunLSTM(darknet_model, hidden_size=hidden_size)
        combined_feature_size = 20 + hidden_size #total num of features of 3 model outputs
        trained_model = GPM2(gun_model, pose_model, combined_feature_size)
    elif model_type == 'GP':
        gun_model = CustomDarknet53(darknet_model)
        combined_feature_size = 20 + 20 #total num of features of 3 model outputs
        trained_model = GP(gun_model, pose_model, combined_feature_size)
    elif model_type == 'GPM-opt':
        gun_model = Gun_Optimized()
        motion_model = motion_analysis.MotionLSTM(hidden_size, lstm_layers)
        combined_feature_size = 20 + 20 + hidden_size #total num of features of 3 model outputs
        trained_model = GPM1(gun_model, pose_model, motion_model, combined_feature_size)
    elif model_type == 'GPM2-opt':
        gun_model = GunLSTM_Optimized(hidden_size, lstm_layers)
        combined_feature_size = 20 + hidden_size #total num of features of 3 model outputs
        trained_model = GPM2(gun_model, pose_model, combined_feature_size)
    else:
        gun_model = Gun_Optimized()
        combined_feature_size = 20 + 20 #total num of features of 3 model outputs
        trained_model = GPM2(gun_model, pose_model, combined_feature_size)

    # LOAD WEIGHTS
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    trained_model.to(device)

    # LOAD VALIDATION DATASET
    val_dataset = Subset(dataset=custom_dataset, indices=val_indices)

    # PREDICT
    incorrect_predictions = []
    video_loader = DataLoader(val_dataset)
    trained_model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for index, video in enumerate(video_loader):
            print(f'Predicting Fold {fold_num+1}...[{index}/{len(video_loader)}]', end='\r')
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
            
            predictions.append(predicted_label.item())
            targets.append(target_label.item())

        print(f'Predicting Fold {fold_num+1}...[{len(video_loader)}/{len(video_loader)}]')
    
    val_confusion_matrix = confusion_matrix(targets, predictions)
    plot_confusion_matrix(val_confusion_matrix, classes=['0', '1'], title=f'{model_type} Test Matrix', path=f'{root_folder}/fold{fold_num+1}/fold{fold_num+1}_val_confusion_matrix.png')

    # SAVE INCORRECT PREDICTIONS
    incorrect_predictions_path = f'{root_folder}/fold{fold_num+1}/TestIncorrectPredictions.txt'
    with open(incorrect_predictions_path, 'w') as file:
        file.write('Incorrect Predictions:\n\n')
        for item in incorrect_predictions:
            # print(item['data_name'] + ':')
            # print(f'  Predicted Label: {item["predicted_label"]}')
            # print(f'  Correct Label: {item["target_label"]}')
            # print()
            file.write(item['data_name'] + ':\n')
            file.write(f'  Predicted Label: {item["predicted_label"]}\n')
            file.write(f'  Correct Label: {item["target_label"]}\n\n')
        print(f'Incorrect Predictions saved to: {incorrect_predictions_path}')