import cv2
import os
import json
import copy
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.body import Body
from src import util
from src.modules import handregion, bodykeypoints, handimage, motion_preprocess
from src.modules.binarypose import BinaryPose
from src.modules.gun_yolo import CustomDarknet53_NoDense
from src.modules.posecnn import poseCNN
from src.modules.gun_yolo import CustomDarknet53, GunLSTM, GunLSTM_Optimized, Gun_Optimized
from src.modules.combined_model import GPM2
from holocron.models import darknet53
import time
import torch
from torchvision import transforms
import torchvision
import numpy as np
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

darknet_model = darknet53(pretrained=True)
darknet_model = CustomDarknet53_NoDense(darknet_model)
darknet_model.to(device)
darknet_model.eval()




dataset_folder = "raw_dataset/dataset/"
video_label = "11"
# video_label = "no_gun_14"

# Path of input video
video_folder = dataset_folder + video_label





# Initialize body estimation model
body_estimation = Body('model/body_pose_model.pth')

# Specify the folder containing the images/frames
image_folder = video_folder

# Get a list of image file names in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
image_files.sort()  # Sort the files to ensure the correct order


# initialize list of hand tensors
window_size = 5

input_image = np.zeros((224,224, 3), dtype=np.uint8)
input_image = transforms.ToTensor()(input_image)
input_image = input_image.unsqueeze(0)

# get hand feature by darknet53
if torch.cuda.is_available():
    input_image = input_image.cuda()

hand_tensors = []
for _ in range (window_size):
    black_gun_feature_tensor = [0] * 1024
    hand_tensors.append(torch.tensor([black_gun_feature_tensor]).to(device))



# Function to load and process an image frame
def process_frame(frame_number):
    print("")
    print("Frame Num: ", frame_number)
    image_file = image_files[frame_number]
    print(f"Processing image: {image_file}")

    # Load the image
    test_image = os.path.join(image_folder, image_file)
    orig_image = cv2.imread(test_image)  # B,G,R order

    orig_image_shape = orig_image.shape[:2]

    # Resize the image
    target_size = (512,512)
    resized_image = cv2.resize(orig_image, target_size)
    resized_image_shape = resized_image.shape[:2]

    # Body pose estimation
    candidate, subset = body_estimation(resized_image)

    # Visualize body pose on the image
    canvas = copy.deepcopy(resized_image)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    # Extract keypoints data (coordinates and confidence scores)
    keypoints_per_frame = {
        'frame_number': frame_number,
        'keypoints': []
    }

    for person_id in range(len(subset)):
        print("Person ID: ", person_id)

        confidence_min = 0.1
        # extract keypoints dictionary (person_id,keypoints)
        keypoints = bodykeypoints.extract_keypoints(person_id, candidate, subset, confidence_min)

        # plot keypoints
        bodykeypoints.plot_keypoints(canvas,keypoints)

        # add keypoints to keypoints_per_frame list
        keypoints_per_frame['keypoints'].append(keypoints)

        # get box coordinates of hand regions
        hand_intersect_threshold = 0.9
        hand_regions = handregion.extract_hand_regions(keypoints, hand_intersect_threshold)
        print("Hand regions of resized image: ", hand_regions)
        

        # draw hand regions on canvas
        handregion.draw_hand_regions(canvas, hand_regions)

        # create and save concatenated hand region image
        hand_image_width = 256
        


        # generate hand region image
        hand_folder = ""
        hand_region_image, _ = handimage.create_hand_image(resized_image, hand_regions, resized_image_shape, hand_image_width, frame_number, hand_folder, save=False)

        # generate binary pose image
        binary_folder = ""
        binary_pose_image, _ = BinaryPose.createBinaryPose(keypoints, frame_number, binary_folder, save=False)

        if hand_region_image is not None and binary_pose_image is not None:
            # gen list of hand tensors
            global darknet_model
            darknet_model.to(device)
            darknet_model.eval()
            if hand_region_image is not None:
                cv2.imshow("hand region image", hand_region_image)

                preprocess = transforms.Compose([ transforms.ToTensor() ])
                input_image = preprocess(hand_region_image)
                input_image = input_image.unsqueeze(0)
                print(input_image)
                print(input_image.shape)
                

                # get hand feature by darknet53
                if torch.cuda.is_available():
                    input_image = input_image.cuda()
                gun_feature_tensor = darknet_model(input_image)
            else:
                gun_feature_tensor = black_gun_feature_tensor

            print(gun_feature_tensor)
            print(gun_feature_tensor.shape)

            global hand_tensors
            print('aaaaaa',hand_tensors)
            # shift hand list by one
            hand_tensors = hand_tensors[1:]
            print('bbbbbb',hand_tensors)
            # appennd current hand tensor to the list
            hand_tensors.append(gun_feature_tensor)
            print('cccccc', hand_tensors)

            gun_data = torch.cat(hand_tensors, dim=0)
            gun_data = gun_data.unsqueeze(0)
            print(gun_data)
            print(gun_data.shape)


            # transform binary pose image to tensor
            numpy_image = transforms.functional.to_tensor(binary_pose_image)
            input_image = torch.tensor(numpy_image).clone().detach()
            input_image = input_image.unsqueeze(0)
            pose_data = input_image
            input('...')

            print("person data:")
            # print("gun data:" , gun_data.size())
            # print("pose data:", pose_data.size())
            # print("gun data:" , gun_data)
            # print("pose data:", pose_data)




            # Change Checkpoint File Location here
            current_directory = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = '/logs/run#45/model/model_epoch_58.pt'
            checkpoint_path = current_directory + checkpoint_path

            checkpoint = torch.load(checkpoint_path) 
            checkpoint = checkpoint['model_state_dict']
            value_to_change = checkpoint.pop('last.weight')
            checkpoint['dense.weight'] = value_to_change
            value_to_change = checkpoint.pop('last.bias')
            checkpoint['dense.bias'] = value_to_change

            hidden_size = 100
            lstm_layers = 1

            # Load Model
            pose_model = poseCNN()
            gun_model = GunLSTM_Optimized(hidden_size, lstm_layers)
            combined_feature_size = 20 + hidden_size #total num of features of 3 model outputs
            trained_model = GPM2(gun_model, pose_model, combined_feature_size)
            trained_model.load_state_dict(checkpoint)
            trained_model.to(device)
            trained_model.eval()
            with torch.no_grad():
                gun_data = gun_data.to(device)
                pose_data = pose_data.to(device)
                combined_output = trained_model(gun_data, pose_data)
                predicted_label = torch.argmax(combined_output)
                predicted_label = predicted_label.item()

            print("PREDICTED LABEL of person ", person_id, " at frame ", frame_number, ": ", predicted_label)


    return canvas

num_frames = len(image_files)

processed_frame_0 = False

# Create a function to update the animation
def update(frame):
    global processed_frame_0
    if frame == 0 and not processed_frame_0:
        # Process frame 0
        current_frame = process_frame(frame)
        plt.imshow(current_frame[:, :, [2, 1, 0]])  # Display the current frame
        plt.axis('off')
        plt.title(f'Frame {frame}')
        processed_frame_0 = True
    else:
        if frame > 0:
            # Process other frames
            plt.clf()  # Clear the previous frame
            current_frame = process_frame(frame)
            plt.imshow(current_frame[:, :, [2, 1, 0]])  # Display the current frame
            plt.axis('off')
            plt.title(f'Frame {frame}')
            if frame == num_frames - 1:
                plt.close()
                cv2.destroyAllWindows()

# Create the animation
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)

# Display the animation
plt.show()
    
