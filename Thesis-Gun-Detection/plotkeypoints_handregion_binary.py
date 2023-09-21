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
from PIL import Image, ImageDraw
import time



# Initialize body estimation model
body_estimation = Body('model/body_pose_model.pth')

# Specify the folder containing the images/frames
image_folder = 'images/dataset/2'

# Get a list of image file names in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
image_files.sort()  # Sort the files to ensure the correct order

# Initialize a list to store the keypoints data
keypoints_data = []

# Function to load and process an image frame
def process_frame(frame_number):
    start_time = time.time()
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
    keypoints_info = {
        'frame_number': frame_number,
        'keypoints': []
    }

    for person_id in range(len(subset)):
        keypoints = {
            'person_id': person_id,
            'keypoints': []
        }
        for kp_id in range(18):  # 18 keypoints for body
            index = int(subset[person_id][kp_id])
            if not(index == -1): #if keypoint is detected, store coordiantes and confidence score
                x = int(candidate[index, 0])
                y = int(candidate[index, 1])
                confidence = candidate[index, 2]
                keypoints['keypoints'].append({
                    'x': x,
                    'y': y,
                    'confidence': confidence
                })
                # Draw keypoints on the image
                cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)  # Red circles for keypoints
            else: #else, put None coordinates and negative confidence score
                keypoints['keypoints'].append({
                    'x': None,
                    'y': None,
                    'confidence': -1
                })


        #Keypoints for hand region extraction
        left_wrist = keypoints['keypoints'][4]
        right_wrist = keypoints['keypoints'][7]
        left_elbow = keypoints['keypoints'][3]
        right_elbow = keypoints['keypoints'][6]
        
        
        # get boxes before overlap checking
        box_left = extract_hand_region(canvas, left_wrist, left_elbow)
        box_right = extract_hand_region(canvas, right_wrist, right_elbow)

        # check overlap
        hand_regions = hand_region_iou(box_left, box_right, 0.9)
        
        # draw hand region bounding boxes
        for hand_region in hand_regions:
            draw_hand_box(canvas, hand_region)
        
        # Create Binary Pose Image
        create_binary_pose(keypoints, frame_number)

        keypoints_info['keypoints'].append(keypoints)

    keypoints_data.append(keypoints_info)

    finish_time = time.time() - start_time
    print(f"Process Time: {finish_time} seconds")

    return canvas

def create_binary_pose(keypoints, frame_number):
    # move the x and y values into a list
    x = []
    y = []
    for k in keypoints['keypoints']:
        x.append(k['x'])
        y.append(k['y'])
    
    # find minimum and max
    x_min, y_min, x_max, y_max = 999, 999, -1, -1
    for a, b in zip(x,y):
        if a is None or b is None: continue
        x_min = min(x_min, a)
        y_min = min(y_min, b)
        x_max = max(x_max, a)
        y_max = max(y_max, b)

    # adjust each coordinates
    for i in range(len(x)):
        if not (x[i] is None or y[i] is None):
            x[i] = int(x[i]) - x_min
            y[i] = int(y[i]) - y_min
    
    # also adjust max values
    x_max = x_max - x_min
    y_max = y_max - y_min
    # get image width and height
    width, height = x_max + 1, y_max + 1
    # create image PIL
    image = Image.new('1', (width, height), 0)
    # draw object from PIL
    draw = ImageDraw.Draw(image)
    # white line
    line_color = 1
    line_thickness = 5
    # custom function to check if keypoint is missing
    def draw_line(x1,y1,x2,y2):
        if not (x1 is None or x2 is None or y1 is None or y2 is None):
            draw.line((x1,y1,x2,y2), fill=line_color, width=line_thickness)

    # nose to neck
    draw_line(x[0], y[0], x[1], y[1]) 
    # left arm
    draw_line(x[5], y[5], x[1], y[1]) 
    draw_line(x[5], y[5], x[6], y[6])
    draw_line(x[7], y[7], x[6], y[6])
    # right arm
    draw_line(x[2], y[2], x[1], y[1])
    draw_line(x[2], y[2], x[3], y[3])
    draw_line(x[4], y[4], x[3], y[3])
    # left leg
    draw_line(x[8], y[8], x[1], y[1])
    draw_line(x[8], y[8], x[9], y[9])
    draw_line(x[10], y[10], x[9], y[9])
    # right leg
    draw_line(x[11], y[11], x[1], y[1])
    draw_line(x[11], y[11], x[12], y[12])
    draw_line(x[13], y[13], x[12], y[12])
    # left face
    draw_line(x[0], y[0], x[14], y[14])
    draw_line(x[16], y[16], x[14], y[14])
    # right face
    draw_line(x[0], y[0], x[16], y[16])
    draw_line(x[17], y[17], x[16], y[16])

    # File Path
    folder_path = f'./images/binary_pose/{image_folder.split("/")[-1]}'
    file_name = f'{folder_path}/pose_{frame_number}.png'

    # Check Directory
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save Image
    image.save(file_name)

    # Print Log
    print(f'Binary Pose Image Save in: {file_name}')
    
# Create a function to update the animation
def update(frame):
    plt.clf()  # Clear the previous frame
    current_frame = process_frame(frame)
    plt.imshow(current_frame[:, :, [2, 1, 0]])  # Display the current frame
    plt.axis('off')
    plt.title(f'Frame {frame}')

# Extract one hand region
def extract_hand_region(canvas, wrist, elbow):
    # Add bounding box of hand area based on wrist and elbow
    if elbow['confidence'] > 0 and wrist['confidence'] > 0:
        # approximate the position of the gun (center) by moving the wrist position further
        extend_ratio = 0.35 # ratio of elbow to wrist distance portion to extend the wrist position
        x_center = wrist['x'] + int((wrist['x'] - elbow['x']) * extend_ratio) 
        y_center = wrist['y'] + int((wrist['y'] - elbow['y']) * extend_ratio)
        # mark center of bounding box
        cv2.circle(canvas, (x_center, y_center), 3, (0, 0, 255), -1)

        radius = max(abs(x_center - elbow['x']),
                    abs(y_center - elbow['y']))
        x_min = x_center - radius
        y_min = y_center - radius
        x_max = x_center + radius
        y_max = y_center + radius
        return [x_min,y_min,x_max,y_max]
    else:
        return None
    
# draw one hand region bounding box if it exist
def draw_hand_box(canvas, box):
    if box is not None:
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2) 

# check if 2 hand region boxes should be combined based on iou, return box/s
def hand_region_iou(boxL, boxR, threshold):
    if boxL is None or boxR is None:
        return [boxL, boxR]
    iou = bb_modified_iou(boxL,boxR)
    if iou > threshold: #if above threshold, combine boxes
        l_x_min = boxL[0]
        l_y_min = boxL[1]
        l_x_max = boxL[2]
        l_y_max = boxL[3]

        r_x_min = boxR[0]
        r_y_min = boxR[1]
        r_x_max = boxR[2]
        r_y_max = boxR[3]

        x_min = min(l_x_min, r_x_min)
        y_min = min(l_y_min, r_y_min)
        x_max = max(l_x_max, r_x_max)
        y_max = max(l_y_max, r_y_max)

        combined_box = [x_min,y_min,x_max,y_max]
        return [combined_box]
    else:
        return [boxL, boxR]

# get intersection over min area of 2 boxes
# modified iou to have higher score for boxes of different sizes
def bb_modified_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area

    # iou = interArea / float(boxAArea + boxBArea - interArea)
	iou = interArea / min(boxAArea, boxBArea)
	# return the intersection over union value
	return iou

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
