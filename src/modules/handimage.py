import cv2
import numpy as np
import os

# Return concatenated image of hand regions
# size: 2 x 1 
def create_hand_image(image, hand_regions, frame_target_size, output_image_width, frame_number, folder_path):
    output_image_size = (output_image_width * 2 , output_image_width)
    hand_image_size = (output_image_width , output_image_width)
    images_list = []

    # if not combined hand region
    if len(hand_regions)>1:
        for hand_region in hand_regions:
            if hand_region is not None:
                # Cropping an image
                cropped_image = image[max(hand_region[1],0):min(hand_region[3],frame_target_size[1]), max(hand_region[0],0):min(hand_region[2],frame_target_size[0])]
                cropped_image = cv2.resize(cropped_image,(hand_image_size))
                images_list.append(cropped_image)
            else:
                black_image = np.zeros((output_image_width, output_image_width, 3), dtype=np.uint8)
                images_list.append(black_image)
    # if combined, put cropped image in the horizontal center  
    else:
        hand_region = hand_regions[0]
        black_image = np.zeros((output_image_width, output_image_width // 2, 3), dtype=np.uint8)
        cropped_image = image[max(hand_region[1],0):min(hand_region[3],frame_target_size[1]), max(hand_region[0],0):min(hand_region[2],frame_target_size[0])]
        cropped_image = cv2.resize(cropped_image,(hand_image_size))

        images_list.append(black_image)
        images_list.append(cropped_image)
        images_list.append(black_image)


    concatenated_cropped = cv2.hconcat(images_list)
    concatenated_cropped = cv2.resize(concatenated_cropped , output_image_size)

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


