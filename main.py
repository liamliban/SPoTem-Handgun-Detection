import cv2
import os
import torch
import shutil
import src.modules.data_creator as data_creator
from src import model
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from src.modules import annotator
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomYolo
from src.modules.combined_model import CombinedModel
from src.modules.combined_model_no_motion import CombinedModelNoMotion

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)

# Choose dataset
dataset_folder = 'images/dataset/'
video_name = "5"

# Folder where data are stored
#   -gun: data/[video_label]/person_[person_id]/hand_image/
#   -pose: data/[video_label]/person_[person_id]/binary_pose/
#   -motion: data/[video_label]/person_[person_id]/motion_keypoints/
data_folder = f'./data/'

# File names of data:
#   -gun: hands_[frame_num].png
#   -pose: pose_[frame_num].png
#   -motion: keypoints_seq.txt

# create:
#   -hand region images (gun), 
#   -binary pose image (pose), 
#   -preprocessed keypoints text file (motion)
display_animation = False
# Path of output video folder
output_folder = data_folder + video_name + "/"

# Clear folder first
if os.path.exists(output_folder):
    for filename in os.listdir(output_folder):
        if os.path.exists(os.path.join(output_folder, filename)):
            shutil.rmtree(output_folder, filename)

num_frames, num_person = data_creator.create_data(dataset_folder, video_name, data_folder, display_animation)


dataset_is_MGD = True
# If dataset if MGD, annotation will be created
if dataset_is_MGD:
    # folder where the generated data by data_creator is stored
    data_folder = "data/"

    # folder where the MGD annotation is stored
    annotation_folder = "images/MGD_annotation/"

    # Create video annotation
    video_labels = annotator.create_MGD_vid_annotation(dataset_folder, data_folder, video_name, output_folder, annotation_folder)

    # Save video annotation
    annotator.save_video_labels_csv(video_labels, output_folder)




# Print or not print features of models
print_gun_feature = False
print_pose_feature = False
print_motion_feature = False

# Load the YOLO model
yolo_model = models.load_model("yolo/config/yolov3.cfg", "yolo/weights/yolov3.weights")

for person_num in range(num_person):
    print("Person id: ", person_num)

    person_folder = output_folder + "person_" + str(person_num) + "/"

    # Get Path of motion data
    motion_keypoints_folder = person_folder + 'motion_keypoints/'
    motion_path = motion_keypoints_folder + "keypoints_seq.txt"
    
    # Check if data file exist
    motion_file_exist = os.path.isfile(motion_path)
    print("\tmotion path exist? ", motion_file_exist , "\tmotion_path: ", motion_path)

    # MOTION FEATURE EXTRACTION Part 1
    if motion_file_exist:
        window_size = 3
        motion_shifted_data = motion_analysis.load_data(motion_path, window_size)
        motion_shifted_data = motion_shifted_data.to(device)
    else:
        raise Exception("Motion keypoints_seq.txt file does not exist!")

    for frame_num  in range(num_frames):
        print("\tFrame num: ", frame_num)

        # Get path of hand data
        hand_image_folder = person_folder + 'hand_image/'
        hand_path = hand_image_folder +  "hands_" + str(frame_num) + ".png"
        
        # Check if data file exist
        hand_file_exist = os.path.isfile(hand_path)
        print("\t\tHand path exist? ", hand_file_exist , "\thand_path: " , hand_path)

        # GUN FEATURE EXTRACTION
        if hand_file_exist:
            print("\t\tExtracting Gun Feature")
            input_size = (416, 416)

            # Load the image as a numpy array
            hand_image = cv2.imread(hand_path)
            hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)

            # Pad the image to 416x416 without distorting it
            original_height, original_width = hand_image.shape[:2]
            padding_height = max(input_size[0] - original_height, 0)
            padding_width = max(input_size[1] - original_width, 0)
            top = padding_height // 2
            bottom = padding_height - top
            left = padding_width // 2
            right = padding_width - left
            padded_img = cv2.copyMakeBorder(hand_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # cv2.imshow("Image Inputted to YOLOv3", padded_img)

            # Convert the image to a PyTorch tensor
            # hand_image = transforms.ToTensor()(hand_image)
            hand_image = transforms.ToTensor()(padded_img)
            # print("padded", hand_image)

            # Add a batch dimension to the tensor
            hand_image = hand_image.unsqueeze(0)
            gun_model_input = hand_image
            print("\t\tInput shape: " , gun_model_input.shape)

            gun_model = CustomYolo(yolo_model)
            gun_model.to(device)
            gun_model.eval()

            if torch.cuda.is_available():
                gun_model_input = gun_model_input.cuda()

            pose_feature = gun_model(gun_model_input)
            yolo_output = pose_feature

            print("\t\tGun Feature Extracted!")
            if print_gun_feature:
                print("\t\tFeature map  for file name: ", hand_path)
                print("\t\t", yolo_output)

            print("\t\tOutput Shape: ", yolo_output.shape)


        print("")




        # Get path of pose data
        binary_pose_folder = person_folder + 'binary_pose/'
        pose_path = binary_pose_folder + "pose_" + str(frame_num) + ".png"

        # Check if data file exist
        pose_file_exist = os.path.isfile(pose_path)
        print("\t\tpose path exist? ", pose_file_exist , "\tpose_path: ", pose_path)

        # POSE FEATURE EXTRACTION
        if pose_file_exist:
            print("\t\tExtracting Pose Feature")
            # Instantiate CNN model for Binary Pose Images
            pose_model = poseCNN()
            preprocess = transforms.Compose([ transforms.ToTensor() ])
            image = cv2.imread(pose_path, cv2.IMREAD_GRAYSCALE)
            input_image = preprocess(image)
            input_image = input_image.unsqueeze(0)
            pose_model_input = input_image
            print("\t\tInput shape: " , pose_model_input.shape)

            if torch.cuda.is_available():
                pose_model_input = pose_model_input.cuda()
            pose_model.to(device)

            pose_feature = pose_model(pose_model_input)

            print("\t\tPose Feature Extracted!")
            if print_pose_feature:
                print(f"\t\tProcessing {pose_path} - Feature Map: {pose_feature}")
            print("\t\tOutput shape: ", pose_feature.shape)
            # USE fmap TO USE FEATURE MAP FROM conv2d_3
            # USE gap TO USE FLATTENED FEATURE MAP FROM GlobalAveragePooling2d_1
        print("")



        # MOTION FEATURE EXTRACTION Part 2
        if motion_file_exist:
            print("\t\tExtracting Motion Feature")
            if frame_num < window_size - 1:
                print("\t\tMotion Analysis: Not enough previous frames. No feature extracted")
            else:
                motion_model_input = motion_shifted_data[frame_num - (window_size - 1)].unsqueeze(0) #get one sequence only
                if torch.cuda.is_available():
                    motion_model_input = motion_model_input.cuda()
                print("\t\tInput shape: " , motion_model_input.shape)

                # Define the model and specify hyperparameters
                input_size = 36
                hidden_size = 64
                num_layers = 1
                output_size = 1

                motion_model = motion_analysis.MotionLSTM()
                motion_model.to(device)

                motion_model.eval()  # Set the model in evaluation mode
                
                with torch.no_grad():
                    motion_feature = motion_model(motion_model_input)

                print("\t\tMotion Feature Extracted!")
                if print_motion_feature:
                    print("\t\t" , motion_feature)
                print("\t\tOutput shape: ", motion_feature.shape)
        print("")


        if hand_file_exist and pose_file_exist and motion_file_exist and not(frame_num < window_size - 1):
            # COMBINED MODEL
            print("\t\tCOMBINATION MODEL")
            combined_feature_size = 20 + 20 + 20 #total num of features of 3 model outputs

            combined_model = CombinedModel(gun_model, pose_model, motion_model, combined_feature_size)
            combined_model.to(device)
            combined_model.eval()
        
            with torch.no_grad():
                combined_output = combined_model(gun_model_input, pose_model_input, motion_model_input)

            print("\t\tCombined Model with Motion Output: ", combined_output)

            # COMBINED MODEL NO MOTION
            combined_2_feature_size = 20 + 20 #total num of features of 2 model outputs

            combined_model_2 = CombinedModelNoMotion(gun_model, pose_model, combined_2_feature_size)
            combined_model_2.to(device)
            combined_model_2.eval()
        
            with torch.no_grad():
                combined_output_2 = combined_model_2(gun_model_input, pose_model_input)

            print("\t\tCombined Model without Motion Output: ", combined_output_2)
        print("")
         




