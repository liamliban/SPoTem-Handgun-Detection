import math
import cv2
import os
import json
import copy
import csv
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.body import Body
from src import util
from src.modules import handregion, bodykeypoints, handimage, motion_preprocess
from src.modules.binarypose import BinaryPose

import torch
import numpy as np
import random

# Set a random seed for reproducibility
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)
os.environ['PYTHONHASHSEED'] = str(12)
torch.cuda.manual_seed_all(12)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

torch.backends.cudnn.deterministic=True

# total number of person in video
total_num_person = 0

# total number of frames in video
num_frames = 0

# prev persons keypoints
prev_persons = None

# create the following data for a video:
#   -hand region images (gun), 
#   -binary pose image (pose), 
#   -preprocessed keypoints text file (motion)
def create_data(dataset_folder, video_label, data_folder, display_animation = False):
    # Path of input video
    video_folder = dataset_folder + video_label

    # Path of output video folder
    output_folder = data_folder + video_label + "/"

    # Initialize body estimation model
    body_estimation = Body('model/body_pose_model.pth')

    # Specify the folder containing the images/frames
    image_folder = video_folder

    # Get a list of image file names in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    # image_files.sort()  # Sort the files to ensure the correct order

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    image_files = sorted(image_files, key=natural_sort_key) # Sort the files to ensure the correct order

    # Initialize a list to store the keypoints data (sequence)
    keypoints_data = []
    normalized_keypoints_data = []

    # Initialize list of hand_regions coordinates 
    # [frame 0 = [person 0 = [hand_regions = [hand_region = [x_min,..., y_max] , ], ], ] , ]
    orig_hand_regions_of_vid = []

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

        num_person = len(subset)

        # Visualize body pose on the image
        canvas = copy.deepcopy(resized_image)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # Extract keypoints data (coordinates and confidence scores)
        keypoints_per_frame = {
            'frame_number': frame_number,
            'keypoints': []
        }
        normalized_keypoints_per_frame = {
            'frame_number': frame_number,
            'keypoints': []
        }

        orig_hand_regions_per_frame = [] #[[person 0] , [person 1] ...]

        # Get all person keypoints in the frame (unarranged)
        unarranged_persons = []
        confidence_min = 0.1
        for person_id in range(num_person):
            keypoints = bodykeypoints.extract_keypoints(person_id, candidate, subset, confidence_min)
            unarranged_persons.append(keypoints)

        
        global prev_persons
        if prev_persons is None:
            # No rearranging in the first frame
            arranged_persons = unarranged_persons
        else:
            arranged_persons = reorder_persons(prev_persons,unarranged_persons)

        

        # update the previous persons for the next iteration
        prev_persons = arranged_persons

        # update max number of person in video
        global total_num_person
        total_num_person = len(arranged_persons)

        for person_id in range(len(arranged_persons)):
            print("Person ID: ", person_id)

            person_folder = output_folder + "person_" + str(person_id) + "/"

            # extract keypoints dictionary (person_id,keypoints)
            # keypoints = bodykeypoints.extract_keypoints(person_id, candidate, subset, confidence_min)
            keypoints = arranged_persons[person_id]

            # add keypoints to keypoints_per_frame list
            keypoints_per_frame['keypoints'].append(keypoints)

            if keypoints is None:
                # add None to normalized_keypoints_per_frame list
                normalized_keypoints_per_frame['keypoints'].append(None)
                # add None to orig_hand_regions_per_frame list
                orig_hand_regions_per_frame.append([None])
            else:    
                # plot keypoints
                bodykeypoints.plot_keypoints(canvas,keypoints)

                # get box coordinates of hand regions
                hand_intersect_threshold = 0.9
                hand_regions = handregion.extract_hand_regions(keypoints, hand_intersect_threshold)
                print("Hand regions of resized image: ", hand_regions)
                
                # get the coordiantes of hand regions for the original image
                orig_hand_regions = handregion.get_orig_hand_regions(orig_image_shape, resized_image_shape, hand_regions)
                print("Hand region of original image: ", orig_hand_regions)

                orig_hand_regions_per_frame.append(orig_hand_regions)

                # draw hand regions on canvas
                handregion.draw_hand_regions(canvas, hand_regions)

                # create and save concatenated hand region image
                hand_image_width = 256
                
                # hand image filename : hands_{frame_number}.png
                hand_folder = person_folder + "hand_image/"
                handregion_image, hand_file_name = handimage.create_hand_image(resized_image, hand_regions, resized_image_shape, hand_image_width, frame_number, hand_folder)
                

                # display the hand region image
                if display_animation and handregion_image is not None:
                    cv2.imshow("hand region image", handregion_image)

                # create and save the binary pose image
                binary_folder = person_folder + "binary_pose/"
                normalized_keypoints, binary_file_name = BinaryPose.createBinaryPose(keypoints, frame_number, binary_folder)

                # add normalized keypoints to normalized_keypoints_per_frame list
                normalized_keypoints_per_frame['keypoints'].append(normalized_keypoints)

        keypoints_data.append(keypoints_per_frame)
        normalized_keypoints_data.append(normalized_keypoints_per_frame)

        orig_hand_regions_of_vid.append(orig_hand_regions_per_frame)

        return canvas

    num_frames = len(image_files)

    processed_frame_0 = False
    if display_animation:
        # Create a function to update the animation
        def update(frame):
            nonlocal processed_frame_0 
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
        if display_animation:
            plt.show()
            
    else:
        for frame in range(num_frames):
        # for frame in range(30):
            process_frame(frame)
        
    # # Save the keypoints data to a JSON file
    # output_json_file = 'keypoints_data.json'
    # with open(output_json_file, 'w') as json_file:
    #     json.dump(keypoints_data, json_file, indent=4)
    # print(f"Keypoints data saved to {output_json_file}")

    # # Save the normalized keypoints data to a JSON file
    # test = normalized_keypoints_data
    # output_json_file = 'normalized_keypoints_data.json'
    # with open(output_json_file, 'w') as json_file:
    #     json.dump(test, json_file, indent=4)
    # print(f"Keypoints data saved to {output_json_file}")



    print("total num person: " , total_num_person)

    for person_id in range(total_num_person):
        # create motion preprocessed data txt file for each person in video
        motion_folder = output_folder + "person_" + str(person_id) + "/motion_keypoints/"
        motion_preprocess.preprocess_data(normalized_keypoints_data, person_id, motion_folder)

        # save hand_regions (original coordinates) sequence of person in a txt file
        handregion.save_hand_regions_txt(output_folder,orig_hand_regions_of_vid)

    return num_frames, total_num_person


def get_num_frames_person(data_folder, video_name):
    csv_file = os.path.join(data_folder, video_name, "video_labels.csv")

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        rows = sum(1 for row in csv_reader)
        file.seek(0)  # Reset the file pointer to the beginning
        header = next(csv_reader)
        columns = len(header)

    num_frames = rows - 1
    num_persons = columns - 1

    return num_frames, num_persons

# Calculate the average distance of two keypoints set
def distance_of_persons(p1, p2):
    kps1 = p1['keypoints']
    kps2 = p2['keypoints']

    sum_distance = 0
    match_ctr = 0

    for i in range(18):
        kp1 = kps1[i]
        kp2 = kps2[i]

        x1 = kp1['x']
        y1 = kp1['y']
        x2 = kp2['x']
        y2 = kp2['y']

        if kp1['confidence'] > 0 and kp2['confidence'] > 0:
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            sum_distance += distance
            match_ctr += 1
    if match_ctr == 0:
        ave_distance = float('inf')
    else:
        ave_distance = sum_distance / match_ctr

    return ave_distance


# Arranged the current list of persons to match the previous list
def reorder_persons(prev_persons, current_persons):
    reordered_persons = [None] * len(prev_persons)
    # initialize list to contain all possible pairs of a prev and current person
    person_pairs = [] # (distance, prev_id, current_id)

    # store all possible pairs
    for prev_id in range(len(prev_persons)):
        for current_id in range(len(current_persons)):
            prev_person = prev_persons[prev_id]
            current_person = current_persons[current_id]
            if prev_person is not None and current_person is not None:
                distance = distance_of_persons(prev_person, current_person)

                pair = (distance, prev_id, current_id)
                person_pairs.append(pair)
        
    # sort the pairs from shortest distance to longest
    person_pairs = sorted(person_pairs, key=lambda x: x[0])

    for pair in person_pairs:
        prev_id = pair[1]
        current_id = pair[2]

        if prev_persons[prev_id] is not None and current_persons[current_id] is not None:
            # Place the current person at the same position as its paired previous person
            reordered_persons[prev_id] = current_persons[current_id]

            # Removed the used persons
            prev_persons[prev_id] = None
            current_persons[current_id] = None
    
    # For all unmatched current persons, add them as a new person id
    for person in current_persons:
        if person is not None:
            reordered_persons.append(person)

    return reordered_persons