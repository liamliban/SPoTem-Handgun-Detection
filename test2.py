import csv
from src.modules import annotator


dataset_folder = "images/dataset/"
data_folder = "data/"
video_name = "5"
output_folder = data_folder + video_name + "/"

annotation_folder = "images/MGD_annotation/"

video_labels = annotator.create_vid_annotation(dataset_folder, data_folder, video_name, output_folder, annotation_folder)

annotator.save_video_labels_csv(video_labels, output_folder)