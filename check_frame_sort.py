import os


video_name = "12"
dataset_folder = "raw_dataset/dataset/"

video_folder = dataset_folder + video_name

# Get a list of image file names in the folder
image_files = [f for f in os.listdir(video_folder) if f.endswith('.jpg')]
image_files.sort()

for index, item in enumerate(image_files):
    print(f"Generated Data Frame: {index} Raw Filename: {item}")
