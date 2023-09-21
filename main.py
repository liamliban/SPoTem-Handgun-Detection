import cv2
import os
import json
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src import model
from src.body import Body
from src import util
from src.modules import handregion, bodykeypoints, binarypose

# Initialize body estimation model
body_estimation = Body('model/body_pose_model.pth')

# Specify the folder containing the images/frames
image_folder = 'images/dataset/2'

# Get a list of image file names in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
image_files.sort()  # Sort the files to ensure the correct order

# Initialize a list to store the keypoints data (sequence)
keypoints_data = []

# Function to load and process an image frame
def process_frame(frame_number):
    image_file = image_files[frame_number]
    print(f"Processing image: {image_file}")

    # Load the image
    test_image = os.path.join(image_folder, image_file)
    oriImg = cv2.imread(test_image)  # B,G,R order

    # Preprocessing:
    # Resize the image to a target size (e.g., 368x368 pixels)
    target_size = (368, 368)
    oriImg = cv2.resize(oriImg, target_size)

    # Body pose estimation
    candidate, subset = body_estimation(oriImg)

    # Visualize body pose on the image
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    # Extract keypoints data (coordinates and confidence scores)
    keypoints_per_frame = {
        'frame_number': frame_number,
        'keypoints': []
    }

    for person_id in range(len(subset)):
        # extract keypoints dictionary (person_id,keypoints)
        keypoints = bodykeypoints.extract_keypoints(person_id, candidate, subset)

        # plot keypoints
        bodykeypoints.plot_keypoints(canvas,keypoints)

        keypoints_per_frame['keypoints'].append(keypoints)

        hand_regions = handregion.extract_hand_regions(keypoints, 0.9)
        handregion.draw_hand_regions(canvas, hand_regions)
        binarypose.create_binary_pose(keypoints, frame_number, image_folder)

        # # cropped_images = []
        # # for hand_region in hand_regions:
        # #     # Cropping an image
        # #     cropped_image = oriImg[max(hand_region[1],0):min(hand_region[3],target_size[0]), max(hand_region[0],0):min(hand_region[2],target_size[0])]
        # #     cropped_image = cv2.resize(cropped_image,(100,100))
        # #     cropped_images.append(cropped_image)
        
        # # concatenated_cropped = cv2.hconcat(cropped_images)
        # # # Display concatenated cropped images
        # # cv2.imshow("cropped", concatenated_cropped)
        
    keypoints_data.append(keypoints_per_frame)

    return canvas

# Create a function to update the animation
def update(frame):
    plt.clf()  # Clear the previous frame
    current_frame = process_frame(frame)
    plt.imshow(current_frame[:, :, [2, 1, 0]])  # Display the current frame
    plt.axis('off')
    plt.title(f'Frame {frame}')

# Create the animation
fig, ax = plt.subplots()
num_frames = len(image_files)
ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)

# Display the animation
plt.show()

# Save the keypoints data to a JSON file
output_json_file = 'keypoints_data.json'
with open(output_json_file, 'w') as json_file:
    json.dump(keypoints_data, json_file, indent=4)

print(f"Keypoints data saved to {output_json_file}")
