import cv2

# Main function
# return:
# [combined_box] or [boxL, boxR]
# each box is [x_min,y_min,x_max,y_max] or None
def extract_hand_regions(keypoints, threshold = 0.9):
    #Keypoints for hand region extraction
    left_wrist = keypoints['keypoints'][4]
    right_wrist = keypoints['keypoints'][7]
    left_elbow = keypoints['keypoints'][3]
    right_elbow = keypoints['keypoints'][6]
    
    
    # get boxes before overlap checking
    box_left = _extract_hand_region(left_wrist, left_elbow)
    box_right = _extract_hand_region(right_wrist, right_elbow)

    # check overlap
    hand_regions = _hand_region_iou(box_left, box_right, threshold)

    return hand_regions

def draw_hand_regions(canvas, hand_regions):
    for hand_region in hand_regions:
        _draw_hand_box(canvas, hand_region)

# draw one hand region bounding box if it exist
def _draw_hand_box(canvas, box):
    if box is not None:
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2) 

# Extract one hand region
def _extract_hand_region(wrist, elbow):
    # Add bounding box of hand area based on wrist and elbow
    if elbow['confidence'] > 0 and wrist['confidence'] > 0:
        # approximate the position of the gun (center) by moving the wrist position further
        extend_ratio = 0.35 # ratio of elbow to wrist distance portion to extend the wrist position
        x_center = wrist['x'] + int((wrist['x'] - elbow['x']) * extend_ratio) 
        y_center = wrist['y'] + int((wrist['y'] - elbow['y']) * extend_ratio)

        radius = max(abs(x_center - elbow['x']),
                    abs(y_center - elbow['y']))
        x_min = x_center - radius
        y_min = y_center - radius
        x_max = x_center + radius
        y_max = y_center + radius
        return [x_min,y_min,x_max,y_max]
    else:
        return None

# check if 2 hand region boxes should be combined based on iou, return box/s
def _hand_region_iou(boxL, boxR, threshold):
    if boxL is None or boxR is None:
        return [boxL, boxR]
    iou = _bb_modified_iou(boxL,boxR)
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
def _bb_modified_iou(boxA, boxB):
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

# Save a text file of the hand_regions sequence
def save_hand_regions_txt(output_folder, hand_regions_of_vid):
    string = str(hand_regions_of_vid)

    hand_regions_txt_path = output_folder + "hand_regions_coords.txt"

    with open(hand_regions_txt_path, 'w') as text_file:
        text_file.write(string + '\n')

    print("Hand regions coordiantes stored in: ", hand_regions_txt_path)