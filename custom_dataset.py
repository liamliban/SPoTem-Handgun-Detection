import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import pandas as pd
import csv

window_size = 3

class CustomGunDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data = []  # A list to store data entries
        self.transform = transform
        
        # Load data entries based on the directory structure
        for data_name in range(8, 9):
            data_dir = os.path.join(root_dir, str(data_name))
            annotation_path = os.path.join(data_dir, 'video_labels.csv')
            
            if os.path.exists(data_dir):
                subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

                for subdir in subdirs:  # Iterate through subdirectories
                    vidx_keypoints = {}  # Initialize vidx_keypoints for each subdir
                    framex_keypoints = {}

                    personx_dir = os.path.join(data_dir, subdir)
                    img_hand_path = os.path.join(personx_dir, "hand_image")
                    img_pose_path = os.path.join(personx_dir, "binary_pose")
                    motion_keypoints_path = os.path.join(personx_dir, "motion_keypoints")

                    # Check if all required directories exist
                    if not os.path.exists(img_hand_path) or not os.path.exists(img_pose_path) or not os.path.exists(motion_keypoints_path):
                        print(f"Skipping subdir {subdir}: Required directories missing.")
                        continue  # Skip to the next subdir
                    
                    else:
                        # Load data entries
                        imgs_hand = [img for img in os.listdir(img_hand_path) if img.endswith(".png")]
                        imgs_pose = [img for img in os.listdir(img_pose_path) if img.endswith(".png")]
                        data_names = []
                    
                        
                        for i in range(len(imgs_hand)):
                            img_path_hand = os.path.join(img_hand_path, imgs_hand[i])
                            img_path_pose = os.path.join(img_pose_path, imgs_pose[i])
                            path_motion = os.path.join(motion_keypoints_path, "keypoints_seq.txt")
                            
                            
                            img_filename = os.path.basename(img_path_hand)
                            frame_num = int(img_filename.split('_')[1].split('.')[0])  # Extract the numeric part before '.png'
                            
                            # Load and transform images
                            gun_frame = transforms.ToTensor()(transforms.ToPILImage()(torchvision.io.read_image(img_path_hand)))
                            pose_frame = transforms.ToTensor()(transforms.ToPILImage()(torchvision.io.read_image(img_path_pose)))
                            
                            
                            sample_name = f"Vid{data_name}_{subdir}_Frame{frame_num}"

                            
                            # Read the CSV file
                            with open(annotation_path, 'r') as csvfile:
                                csv_reader = csv.reader(csvfile)
                                header = next(csv_reader) 
                                
                                if subdir in header:
                                    column_index = header.index(subdir)
                                    row_index = frame_num  # CSV row numbers are 0-based

                                    # Iterate through the CSV rows
                                    for i, row in enumerate(csv_reader):
                                        if i == row_index:
                                            framex_label = row[column_index]
                                            # Check if the value is empty or missing, and set it to "null" if so
                                            if not framex_label:
                                                framex_label = "null"
                                else:
                                    print(f"No match found in CSV file for {subdir}")
                            
                            # read motion kps txt file
                            with open(path_motion, 'r') as file:
                                for j, line in enumerate(file, start=0):
                                    values = [float(val) for val in line.strip().split(',')]
                                    vidx_keypoints[j] = values
                                
                            
                            # Conditionally check the counters and create data_entry if they are equal
                            keypoints_set_ctr = len(vidx_keypoints)
                            hand_imgs_ctr = len(imgs_hand)
                            pose_imgs_ctr = len(imgs_pose)
                            
                            
                                
                            if hand_imgs_ctr == pose_imgs_ctr:
                                if frame_num >= window_size-1:
                                    framex_keypoints = {i: torch.tensor(vidx_keypoints[frame_num-i]) for i in range(window_size)}  
                                    data_entry = {
                                        "data_name": sample_name,
                                        "gun_frame": gun_frame,
                                        "pose_frame": pose_frame,
                                        "motion_kps": framex_keypoints,
                                        "label": framex_label
                                    }
                                    self.data.append(data_entry)
                            else:
                                print(f"Skipping {sample_name}: Counters not equal.")
                                continue  # Continue to the next image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_entry = self.data[idx]
        data_name = data_entry["data_name"]
        gun_frame = data_entry["gun_frame"]
        pose_frame = data_entry["pose_frame"]
        motion_kps = data_entry.get("motion_kps")
        frame_label = data_entry["label"]

custom_dataset = CustomGunDataset(root_dir='data')

for idx, data_entry in enumerate(custom_dataset.data):
    print(f"Data Entry {idx + 1}:")
    print("Data Name:", data_entry["data_name"])
    print("Gun Frame Shape:", data_entry["gun_frame"].shape)
    print("Pose Frame Shape:", data_entry["pose_frame"].shape)
    print("Label:", data_entry["label"])
    print("Motion Keypoints:")
    for key, values in data_entry["motion_kps"].items():
        print(f"kps_{key}: {values}")
    
    print()
