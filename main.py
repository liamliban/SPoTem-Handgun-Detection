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
from src.modules import handregion, bodykeypoints, handimage
from yolo.pytorchyolo import detect, models
import torchvision.transforms as transforms
from src.modules.binarypose import BinaryPose
from src.modules.posecnn import poseCNN

# Initialize body estimation model
body_estimation = Body('model/body_pose_model.pth')

# Load the YOLO model
model = models.load_model("yolov3.cfg", "yolov3.weights")

# Specify the folder containing the images/frames
image_folder = 'images/dataset/11'

# Get a list of image file names in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
image_files.sort()  # Sort the files to ensure the correct order

# Initialize a list to store the keypoints data (sequence)
keypoints_data = []
normalized_keypoints_data = []

# Function to load and process an image frame
def process_frame(frame_number):
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
        handregion_image, hand_file_name = handimage.create_hand_image(resized_image, hand_regions, target_size, hand_image_width, frame_number, person_id, image_folder)

        # Load the image as a numpy array
        img = cv2.imread(hand_file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, target_size)

        # Convert the image to a PyTorch tensor
        img = transforms.ToTensor()(img)

        # Add a batch dimension to the tensor
        img = img.unsqueeze(0)

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
        output = model(img.cuda())

        # Print the conv_81 layer activation
        print("CONV 81 LAYER FOR FILE NAME: ", hand_file_name)
        print(activation['conv_81'])
        
        # display the hand region image
        cv2.imshow("hand region image", handregion_image)

        # create and save the binary pose image
        normalized_keypoints, binary_file_name = BinaryPose.createBinaryPose(keypoints, frame_number, image_folder)

        # add normalized keypoints to normalized_keypoints_per_frame list
        normalized_keypoints_per_frame['keypoints'].append(normalized_keypoints)

        if binary_file_name is not None:
            # Instantiate CNN model for Binary Pose Images
            cnn = poseCNN()
            preprocess = transforms.Compose([ transforms.ToTensor() ])
            image = cv2.imread(binary_file_name, cv2.IMREAD_GRAYSCALE)
            input_image = preprocess(image)
            input_image = input_image.unsqueeze(0)
            fmap, gap = cnn(input_image)

            # PRINT OUT FEATURE MAP TO TEST IF READING THE RIGHT FILE. VERY LENGTHY SO COMMENT OUT IF NOT NEEDED
            print(f"Processing {binary_file_name} - Conv2d_3 Feature Map: {fmap}, GAP Feature Map: {gap}")

            # USE fmap TO USE FEATURE MAP FROM conv2d_3
            # USE gap TO USE FLATTENED FEATURE MAP FROM GlobalAveragePooling2d_1


    keypoints_data.append(keypoints_per_frame)
    normalized_keypoints_data.append(normalized_keypoints_per_frame)

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

# Save the normalized keypoints data to a JSON file
output_json_file = 'normalized_keypoints_data.json'
with open(output_json_file, 'w') as json_file:
    json.dump(normalized_keypoints_data, json_file, indent=4)

print(f"Keypoints data saved to {output_json_file}")
