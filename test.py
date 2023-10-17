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
from src.modules import handregion, bodykeypoints, binarypose, handimage

# Initialize body estimation model
body_estimation = Body('model/body_pose_model.pth')

# Specify the folder containing the images/frames
image_folder = 'images/dataset/2'

# Get a list of image file names in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
image_files.sort()  # Sort the files to ensure the correct order

# Initialize a list to store the keypoints data (sequence)
keypoints_data = []

# Load YoloV3
# Using Weights and CFG from https://github.com/Manish8798/Weapon-Detection-with-yolov3 ** NOT GOOD **
#yolo_net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")

# Using Weights and CFG from https://github.com/pjreddie/darknet ** SEEMS TO RUN BETTER **
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
yolo_classes = ["Weapon"]

# Weapon Detection Heavily taken from https://github.com/Manish8798/Weapon-Detection-with-yolov3
def detect_weapon(image, width, height):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    yolo_outs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []
    
    for out in yolo_outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Check if class_id is for "Weapon"
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if indexes.any():
        return True  # Weapon detected
    else:
        return False  # No weapon detected
    

# Function to load and process an image frame
def process_frame(frame_number):
    image_file = image_files[frame_number]
    print(f"Processing image: {image_file}")

    # Load the image
    test_image = os.path.join(image_folder, image_file)
    orig_image = cv2.imread(test_image)  # B,G,R order

    # Preprocessing:
    # Resize the image to a target size (e.g., 368x368 pixels)
    target_size = (368, 368)
    resized_image = cv2.resize(orig_image, target_size)

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
        # extract keypoints dictionary (person_id,keypoints)
        keypoints = bodykeypoints.extract_keypoints(person_id, candidate, subset)

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
        handregion_image = handimage.create_hand_image(resized_image, hand_regions, target_size, hand_image_width, frame_number, image_folder)

        # display the hand region image
        cv2.imshow("hand region image", handregion_image)

        # create and save the binary pose image
        binarypose.create_binary_pose(keypoints, frame_number, image_folder)


    keypoints_data.append(keypoints_per_frame)

     # Detect weapon on hand image
    weapon_detected = detect_weapon(handregion_image, resized_image.shape[1], resized_image.shape[0])

    # Print the result
    if weapon_detected:
        print(f"Weapon detected on frame {frame_number}")
    else:
        print(f"Weapon NOT detected on frame {frame_number}")

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
