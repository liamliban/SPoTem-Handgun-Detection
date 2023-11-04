import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import pandas as pd

window_size = 3

class CustomGunDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.label = pd.read_csv(csv_file)
        self.data = []  # A list to store data entries
        self.transform = transform
        

        # Load data entries based on the directory structure
        for data_name in range(4, 6):
            data_dir = os.path.join(root_dir, str(data_name))
            if os.path.exists(data_dir):
                subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

                for subdir in subdirs:  # Iterate through subdirectories
                    images_dir = os.path.join(data_dir, subdir)
                    img_hand_path = os.path.join(images_dir, "hand_image")
                    img_pose_path = os.path.join(images_dir, "binary_pose")
                    motion_keypoints_path = os.path.join(images_dir, "motion_keypoints")

                    # Check if all required directories exist
                    if not os.path.exists(img_hand_path) or not os.path.exists(img_pose_path) or not os.path.exists(motion_keypoints_path):
                        # If any of the directories is missing, skip this subdirectory
                        continue

                    # Load data entries
                    imgs_hand = [img for img in os.listdir(img_hand_path) if img.endswith(".png")]
                    imgs_pose = [img for img in os.listdir(img_pose_path) if img.endswith(".png")]
                    data_names = []

                    for i in range(len(imgs_hand)):
                        img_path_hand = os.path.join(img_hand_path, imgs_hand[i])
                        img_path_pose = os.path.join(img_pose_path, imgs_pose[i])
                        img_path_motion = os.path.join(motion_keypoints_path, "keypoints_seq.txt")

                        # Create data_name based on your specifications
                        frame_num = i+1
                        sample_name = f"Vid{data_name}_{subdir}_Frame{i+1}"
                        data_names.append(sample_name)

                        # Load and transform images
                        gun_frame = transforms.ToTensor()(transforms.ToPILImage()(torchvision.io.read_image(img_path_hand)))
                        pose_frame = transforms.ToTensor()(transforms.ToPILImage()(torchvision.io.read_image(img_path_pose)))

                        # Initialize an empty dictionary to store the data
                        vidx_keypoints = {}
                        framex_keypoints = {}

                        # Read the lines from the text file and store them in the dictionary
                        with open(img_path_motion, 'r') as file:
                            for j, line in enumerate(file, start=1):
                                values = [float(val) for val in line.strip().split(',')]
                                vidx_keypoints[j] = values
                        
                        if frame_num < window_size:
    # Skip to the next frame if within the window_size
                            continue
                        else:
                            # Locate the key-value sets based on frame_num and window_size
                            framex_keypoints = {i: torch.tensor(vidx_keypoints[frame_num - i]) for i in range(window_size)}

                            
                        # Conditionally check the counters and create data_entry if they are equal
                        keypoints_set_ctr = len(vidx_keypoints)
                        hand_imgs_ctr = len(imgs_hand)
                        pose_imgs_ctr = len(imgs_pose)

                        if keypoints_set_ctr == hand_imgs_ctr == pose_imgs_ctr:
                            # All counters are equal, create data_entry
                            data_entry = {
                                "data_name": sample_name,
                                "gun_frame": gun_frame,  # Store the loaded image data
                                "pose_frame": pose_frame,  # Store the loaded image data
                                "motion_kps": framex_keypoints  # Store motion keypoints data
                            }
                            self.data.append(data_entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_entry = self.data[idx]
        data_name = data_entry["data_name"]
        gun_frame = data_entry["gun_frame"]  # Get the loaded image data
        pose_frame = data_entry["pose_frame"]  # Get the loaded image data
        motion_kps = data_entry.get("motion_kps")  # Retrieve motion keypoints if it exists

custom_dataset = CustomGunDataset(root_dir='data', csv_file='images/annotations.csv')

# Loop through the data entries and print them
for idx, data_entry in enumerate(custom_dataset.data):
    print(f"Data Entry {idx + 1}:")
    print("Data Name:", data_entry["data_name"])
    print("Gun Frame Shape:", data_entry["gun_frame"].shape)
    print("Pose Frame Shape:", data_entry["pose_frame"].shape)
    
    print("Motion Keypoints:")
    for key, values in data_entry["motion_kps"].items():
        print(f"kps_{key}: {values}")
    
    print()