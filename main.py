import cv2
import os
import torch
import src.modules.data_creator as data_creator
from src import model
from src.modules import motion_analysis
from yolo.pytorchyolo import models
import torchvision.transforms as transforms
from src.modules.posecnn import poseCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device: " , device)

# Choose dataset
dataset_folder = 'images/dataset/'
video_label = "8"

# Folder where data are stored
#   -gun: data/[video_label]/hand_image/[person_id]/
#   -pose: data/[video_label]/binary_pose/[person_id]/
#   -motion: data/[video_label]/motion_keypoints/[person_id]/
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
num_frames, num_person = data_creator.create_data(dataset_folder, video_label, data_folder, display_animation)



# Folders of data
hand_image_folder = data_folder + video_label + '/hand_image/'
binary_pose_folder = data_folder + video_label + '/binary_pose/'
motion_keypoints_folder = data_folder + video_label + '/motion_keypoints/'

# Print or not print features of models
print_gun_feature = False
print_pose_feature = False
print_motion_feature = False

# Load the YOLO model
model = models.load_model("yolo/config/yolov3.cfg", "yolo/weights/yolov3.weights")

for person_num in range(num_person):
    print("Person id: ", person_num)

    # Get Path of motion data
    motion_path = motion_keypoints_folder + str(person_num) + "/" + "keypoints_seq.txt"
    
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
        hand_path = hand_image_folder + str(person_num) + "/" + "hands_" + str(frame_num) + ".png"
        
        # Check if data file exist
        hand_file_exist = os.path.isfile(hand_path)
        print("\t\tHand path exist? ", hand_file_exist , "\thand_path: " , hand_path)

        if hand_file_exist:
            # GUN FEATURE EXTRACTION
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
            hand_image = transforms.ToTensor()(hand_image)

            # Add a batch dimension to the tensor
            hand_image = hand_image.unsqueeze(0)

            # Set the model to evaluation mode
            model.eval()

            # Get the conv_81 layer
            conv81 = model.module_list[81]

            # Create a dictionary to store the activations
            activation = {}

            # Define a forward hook to capture the activation of the conv_81 layer
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook

            # Register the forward hook on the conv_81 layer
            conv81.register_forward_hook(get_activation('conv_81'))

            # Forward pass the image through the model
            if torch.cuda.is_available():
                hand_image = hand_image.cuda()
            output = model(hand_image)

            # Print the conv_81 layer activation
            print("\t\tGun Feature Extracted!")
            if print_gun_feature:
                print("\t\tCONV 81 LAYER FOR FILE NAME: ", hand_path)
                print("\t\t", activation['conv_81'])
        print("")




        # Get path of pose data
        pose_path = binary_pose_folder + str(person_num) + "/" + "pose_" + str(frame_num) + ".png"

        # Check if data file exist
        pose_file_exist = os.path.isfile(pose_path)
        print("\t\tpose path exist? ", pose_file_exist , "\tpose_path: ", pose_path)

        if pose_file_exist:
            # POSE FEATURE EXTRACTION
            print("\t\tExtracting Pose Feature")
            # Instantiate CNN model for Binary Pose Images
            cnn = poseCNN()
            preprocess = transforms.Compose([ transforms.ToTensor() ])
            image = cv2.imread(pose_path, cv2.IMREAD_GRAYSCALE)
            input_image = preprocess(image)
            input_image = input_image.unsqueeze(0)
            fmap, gap = cnn(input_image)

            print("\t\tPose Feature Extracted!")
            if print_pose_feature:
                print(f"\t\tProcessing {pose_path} - Conv2d_3 Feature Map: {fmap}, GAP Feature Map: {gap}")

            # USE fmap TO USE FEATURE MAP FROM conv2d_3
            # USE gap TO USE FLATTENED FEATURE MAP FROM GlobalAveragePooling2d_1
        print("")



        # MOTION FEATURE EXTRACTION Part 2
        if motion_file_exist:
            print("\t\tExtracting Motion Feature")
            if frame_num < window_size - 1:
                print("\t\tMotion Analysis: Not enough previous frames. No feature extracted")
            else:
                motion_shifted_data_frame = motion_shifted_data[frame_num - (window_size - 1)].unsqueeze(0) #get one sequence only

                # Define the model and specify hyperparameters
                input_size = 36
                hidden_size = 64
                num_layers = 1
                output_size = 1

                motion_model = motion_analysis.MotionLSTM(input_size, hidden_size, num_layers, output_size)
                motion_model.to(device)

                motion_model.eval()  # Set the model in evaluation mode

                with torch.no_grad():
                    motion_feature = motion_model(motion_shifted_data_frame)

                print("\t\tMotion Feature Extracted!")
                if print_motion_feature:
                    print("\t\t" , motion_feature)
        print("")




