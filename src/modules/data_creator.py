import cv2
import os
import json
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.body import Body
from src import util
from src.modules import handregion, bodykeypoints, handimage, motion_preprocess
from src.modules.binarypose import BinaryPose

# total number of person in video
total_num_person = 0

# total number of frames in video
num_frames = 0

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
    image_files.sort()  # Sort the files to ensure the correct order

    # Initialize a list to store the keypoints data (sequence)
    keypoints_data = []
    normalized_keypoints_data = []

    # Function to load and process an image frame
    def process_frame(frame_number):
        print("Frame Num: ", frame_number)
        image_file = image_files[frame_number]
        print(f"Processing image: {image_file}")

        # Load the image
        test_image = os.path.join(image_folder, image_file)
        orig_image = cv2.imread(test_image)  # B,G,R order

        # Preprocessing:
        # Resize the image to a target size (e.g., 368x368 pixels)
        target_size = (416, 416)
        resized_image = cv2.resize(orig_image, target_size)

        # Body pose estimation
        candidate, subset = body_estimation(resized_image)

        # update max number of person in video
        global total_num_person
        total_num_person = max(total_num_person, len(subset))

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

            # draw hand regions on canvas
            handregion.draw_hand_regions(canvas, hand_regions)

            # create and save concatenated hand region image
            hand_image_width = 256
            
            # hand image filename : hands_{frame_number}_{person_id}.png
            hand_folder = output_folder + "hand_image/"
            handregion_image, hand_file_name = handimage.create_hand_image(resized_image, hand_regions, target_size, hand_image_width, frame_number, person_id, hand_folder)
            

            # display the hand region image
            if display_animation:
                cv2.imshow("hand region image", handregion_image)

            # create and save the binary pose image
            binary_folder = output_folder + "binary_pose/"
            normalized_keypoints, binary_file_name = BinaryPose.createBinaryPose(keypoints, frame_number, binary_folder)

            # add normalized keypoints to normalized_keypoints_per_frame list
            normalized_keypoints_per_frame['keypoints'].append(normalized_keypoints)

        keypoints_data.append(keypoints_per_frame)
        normalized_keypoints_data.append(normalized_keypoints_per_frame)

        return canvas

    num_frames = len(image_files)
    if display_animation:
        # Create a function to update the animation
        def update(frame):
            plt.clf()  # Clear the previous frame
            current_frame = process_frame(frame)
            plt.imshow(current_frame[:, :, [2, 1, 0]])  # Display the current frame
            plt.axis('off')
            plt.title(f'Frame {frame}')

        # Create the animation
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)

        # Display the animation
        if display_animation:
            plt.show()
    else:
        for frame in range(num_frames):
            process_frame(frame)
        

    # # Save the keypoints data to a JSON file
    # output_json_file = 'keypoints_data.json'
    # with open(output_json_file, 'w') as json_file:
    #     json.dump(keypoints_data, json_file, indent=4)

    # print(f"Keypoints data saved to {output_json_file}")

    # # Save the normalized keypoints data to a JSON file
    # output_json_file = 'normalized_keypoints_data.json'
    # with open(output_json_file, 'w') as json_file:
    #     json.dump(normalized_keypoints_data, json_file, indent=4)

    # print(f"Keypoints data saved to {output_json_file}")

    print("total num person: " , total_num_person)

    # print("keypoints test")
    # for frame in keypoints_data:
    #     print("\tframe num : ", frame.get("frame_number"))
    #     for person in frame.get("keypoints"):
    #         print("\t\tperson id : ", person.get("person_id"))
    #         print("\t\tkps length : ", len(person.get("keypoints")))

    # print("normalized keypoints test")
    # for frame in normalized_keypoints_data:
    #     print("\tframe num : ", frame.get("frame_number"))
    #     for person in frame.get("keypoints"):
    #         print("\t\tperson id : ", person.get("person_id"))
    #         print("\t\tkps length : ", len(person.get("keypoints")))

    # create motion preprocessed data txt file for each person in video
    motion_folder = output_folder + "motion_keypoints/"
    for person_id in range(total_num_person):
        motion_preprocess.preprocess_data(normalized_keypoints_data, person_id, motion_folder)

    return num_frames, total_num_person


