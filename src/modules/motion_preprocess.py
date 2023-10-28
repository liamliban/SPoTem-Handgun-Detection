import json
import os

class PersonIDNotFoundError(Exception):
    pass

# Create file keypoints_seq_[person_id].txt which stores keypoints of frame per line with format
# x0, y0, x1, y1 .... x17, y17
def preprocess_data(keypoints_file_path, video_label, person_id):  
    # try:
        null_value = 999
        keypoints_sequence = _get_normalized(keypoints_file_path, person_id, null_value)
        # for index, keypoint_set in enumerate(keypoints_sequence, start=1):
        #     print(f"Frame {index}: {keypoint_set}")

        file_path = _save_keypoints(keypoints_sequence, video_label, person_id)
        return file_path

    # except PersonIDNotFoundError as e:
    #     print(e)

    # except FileNotFoundError as e:
    #     print(e)

# gets the normalized keypoints in a json file and return a list (per frame) of a list of body keypoints (x0, y0, x1, y1 .... x17, y17) 
def _get_normalized(data_file_path, person_id, null_value):
    keypoint_sequences = []

    try:
        with open(data_file_path, 'r') as json_file:
            data = json.load(json_file)

            person_found = False  # Flag to track if the person ID is found

            for frame_data in data:
                keypoints = frame_data.get("keypoints")

                for person_data in keypoints:
                    if person_data.get("person_id") == person_id:
                        person_keypoints = person_data.get("keypoints")
                        keypoint_set = []

                        for keypoint in person_keypoints:
                            x = keypoint.get("x")
                            y = keypoint.get("y")

                            if x is None:
                                x = null_value

                            if y is None:
                                y = null_value

                            keypoint_set.extend([x, y])

                        keypoint_sequences.append(keypoint_set)

                        person_found = True

            if not person_found:
                raise PersonIDNotFoundError(f"Error: Person ID {person_id} not found in the JSON data")

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {data_file_path}")

    return keypoint_sequences

# takes a keypoints_sequence and save it into a text file
def _save_keypoints(keypoints_sequence, video_label, person_id):
    # File Path
    folder_path = f'./data/motion_keypoints/{video_label.split("/")[-1]}'
    file_path = f'{folder_path}/keypoints_seq_{person_id}.txt'

    # Check Directory
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(file_path, "w") as text_file:
        for i in range(len(keypoints_sequence)):
            for j in range(36):
                text_file.write('{}'.format(keypoints_sequence[i][j]))
                if j < 35:
                    text_file.write(',')
            text_file.write('\n')

    # Print Log
    print(f'Keypoints sequence stored in: {file_path}')
    return file_path

# Example usage:
# keypoints_file_path = "normalized_keypoints_data.json"
# video_label = "test"
# person_id = 1
# null_value = 999

# preprocess_data(keypoints_file_path, video_label, person_id)
