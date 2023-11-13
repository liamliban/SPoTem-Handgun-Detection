import cv2
import numpy as np
import os
import torch
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

# Return concatenated image of hand regions
# size: 2 x 1 
def create_hand_image(image, hand_regions, frame_image_shape, output_image_width, frame_number, folder_path):
    output_image_size = (output_image_width * 2 , output_image_width)
    hand_image_size = (output_image_width , output_image_width)
    images_list = []

    # if not combined hand region
    if len(hand_regions)>1:
        for hand_region in hand_regions:
            if hand_region is not None:
                # Cropping an image
                cropped_image = image[max(hand_region[1],0):min(hand_region[3],frame_image_shape[1]), max(hand_region[0],0):min(hand_region[2],frame_image_shape[0])]
                if not cropped_image.size == 0:
                    # cropped_image = cv2.resize(cropped_image,(hand_image_size))
                    cropped_image = make_image_square(cropped_image, output_image_width)
                else:
                    cropped_image = np.zeros((output_image_width, output_image_width, 3), dtype=np.uint8)
                images_list.append(cropped_image)
            else:
                black_image = np.zeros((output_image_width, output_image_width, 3), dtype=np.uint8)
                images_list.append(black_image)
    # if combined, put cropped image in the horizontal center  
    else:
        hand_region = hand_regions[0]
        black_image = np.zeros((output_image_width, output_image_width // 2, 3), dtype=np.uint8)
        cropped_image = image[max(hand_region[1],0):min(hand_region[3],frame_image_shape[1]), max(hand_region[0],0):min(hand_region[2],frame_image_shape[0])]
        if not cropped_image.size == 0:
            # cropped_image = cv2.resize(cropped_image,(hand_image_size))
            cropped_image = make_image_square(cropped_image, output_image_width)
        else:
            cropped_image = np.zeros((output_image_width, output_image_width, 3), dtype=np.uint8)

        images_list.append(black_image)
        images_list.append(cropped_image)
        images_list.append(black_image)


    concatenated_cropped = cv2.hconcat(images_list)
    concatenated_cropped = cv2.resize(concatenated_cropped , output_image_size)

    image_is_blank = True if cv2.countNonZero(cv2.cvtColor(concatenated_cropped, cv2.COLOR_BGR2GRAY)) == 0 else False

    if image_is_blank:
        return None, None

    # File Path
    file_name = f'{folder_path}/hands_{frame_number}.png'

    # Check Directory
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save Image
    cv2.imwrite(file_name, concatenated_cropped)

    # Print Log
    print(f'Hands Image Save in: {file_name}')

    return concatenated_cropped, file_name

def make_image_square(image, size):
    height, width, _ = image.shape

    # Determine the size of the square
    square_size = max(height, width)

    # Create a new square canvas with the desired size
    square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)

    # Calculate the position to paste the original image at the center
    x_offset = (square_size - width) // 2
    y_offset = (square_size - height) // 2

    # Paste the original image onto the square canvas
    square_image[y_offset:y_offset + height, x_offset:x_offset + width] = image

    # Resize the square image to the desired size
    square_image = cv2.resize(square_image, (size, size))

    return square_image



# Save images individually
def create_hand_image_single(image, hand_regions, frame_image_shape, output_image_width, frame_number, folder_path):

    # # if hands are combined, double the regions
    # if len(hand_regions) == 1:
    #     hand_regions.append(hand_regions[0])

    # for all hand regions, cropped image
    for hand_num in range(len(hand_regions)):
        hand_region = hand_regions[hand_num]
        if hand_region is not None:
            # Cropping an image
            cropped_image = image[max(hand_region[1],0):min(hand_region[3],frame_image_shape[1]), max(hand_region[0],0):min(hand_region[2],frame_image_shape[0])]
            if not cropped_image.size == 0:
                # cropped_image = cv2.resize(cropped_image,(hand_image_size))
                cropped_image = make_image_square(cropped_image, output_image_width)

                # if hands are combined
                if len(hand_regions) == 1:
                    # File Path
                    file_name = f'{folder_path}/hands_{frame_number}_c.png'
                else:
                    # File Path
                    file_name = f'{folder_path}/hands_{frame_number}_{hand_num}.png'

                # Check Directory
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # Save Image
                cv2.imwrite(file_name, cropped_image)

                # Print Log
                print(f'Hands Image Save in: {file_name}')





