import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from src.modules.gun_yolo import CustomDarknet53_NoDense
from holocron.models import darknet53

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def _get_hand_images_names(hand_image_folder):
    png_file_names = []
    for root, dirs, files in os.walk(hand_image_folder):
        for file in files:
            if file.lower().endswith('.png'):
                # Remove '.png' extension
                png_file_names.append(os.path.splitext(file)[0])
    return png_file_names

def _get_hand_image(hand_path):
    # Load the image as a numpy array
    hand_image = cv2.imread(hand_path)
    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)

    original_height, original_width = hand_image.shape[:2]

    target_width = 416

    # Calculate the scaling factor for the width to make it 416
    scale_factor = target_width / original_width
    scaled_image = cv2.resize(hand_image, (target_width, int(original_height * scale_factor)))

    # Calculate the necessary padding for height
    original_height, original_width = scaled_image.shape[:2]
    target_height = 416

    padding_height = max(target_height - original_height, 0)

    # Calculate the top and bottom padding dimensions
    top = padding_height // 2
    bottom = padding_height - top

    # Pad the image to achieve the final size of 416x416
    padded_image = cv2.copyMakeBorder(scaled_image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


    # TEMPORARY: resize the image to 224
    padded_image = cv2.resize(padded_image, (224,224))

    return padded_image


darknet_model = darknet53(pretrained=True)
gun_model = CustomDarknet53_NoDense(darknet_model)
gun_model.to(device)
gun_model.eval()

def save_feature_tensor_of_black(data_folder, h, w):
    image = np.zeros((h,w, 3), dtype=np.uint8)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)

    if torch.cuda.is_available():
        image = image.cuda()


    feature_tensor = gun_model(image)
    # print(feature_tensor)
    tensor_path = os.path.join(data_folder,'darknet_black_feature_tensor.pt') 
    torch.save(feature_tensor,tensor_path)
    print("Black Feature tensor stored in : ", tensor_path)

def _save_one_feature_tensor(hand_image_folder, image_name):
    hand_path = os.path.join(hand_image_folder,image_name + '.png')
    image = _get_hand_image(hand_path)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)

    if torch.cuda.is_available():
        image = image.cuda()


    feature_tensor = gun_model(image)
    # print(feature_tensor)
    tensor_path = os.path.join(hand_image_folder,image_name + '.pt') 
    torch.save(feature_tensor,tensor_path)
    print("Hand Image Feature tensor stored in : ", tensor_path)


def _save_person_feature_tensors(hand_image_folder):
 hand_images_names = _get_hand_images_names(hand_image_folder)

 for image_name in hand_images_names:
    _save_one_feature_tensor(hand_image_folder,image_name)

def _save_video_feature_tensors(video_folder):
    person_folders = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]

    for person_folder in person_folders:
        print("Checking person folder : " , person_folder)
        hand_image_folder = os.path.join(person_folder,'hand_image')
        _save_person_feature_tensors(hand_image_folder)

def save_all_hand_feature_tensors(data_folder):
    video_folders = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]

    for video_folder in video_folders:
        print("Checking video folder : " , video_folder)
        _save_video_feature_tensors(video_folder)

data_folder = 'data'

# Apply darknet to all hand images and store the feature tensors for faster training
save_all_hand_feature_tensors(data_folder)

# Create a black image and apply darknet and store the feature tensor (used in dataset)
save_feature_tensor_of_black(data_folder,224,224)

# print(torch.load('data/darknet_black_tensor.pt'))