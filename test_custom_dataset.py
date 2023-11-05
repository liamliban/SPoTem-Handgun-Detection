from src.modules.custom_dataset import CustomGunDataset

# Specifiy the video. = None if all videos
# video = None
video = '8'

custom_dataset = CustomGunDataset(root_dir='data', video=video)

print("Dataset number of samples: ", len(custom_dataset))
for idx, data_entry in enumerate(custom_dataset.data):
    print(f"Data Entry {idx + 1}:")
    print("Data Name:", data_entry["data_name"])
    print("Gun Frame Shape:", data_entry["gun_frame"].shape)
    print("Pose Frame Shape:", data_entry["pose_frame"].shape)
    print("Label:", data_entry["label"])
    print("Motion Keypoints:") 
    print(data_entry["motion_kps"])




