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

        orig_hand_regions_per_frame = [] #[[person 0] , [person 1] ...]

        for person_id in range(len(subset)):
            print("Person ID: ", person_id)

            person_folder = output_folder + "person_" + str(person_id) + "/"

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
            if display_animation:
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