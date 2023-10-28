import numpy as np
import torch
import torch.nn as nn

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load the and shift data into a numpy array based on window_size, returns list of sequence of data
def load_data(file_path, window_size):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if window_size > len(lines):
            raise Exception("Timestep should not be greater than num of samples")
        for i in range(window_size, len(lines) + 1):
            sequence = []
            for j in range(i - window_size, i):
                line = lines[j].strip().split(',')
                sequence.append([float(val) for val in line])
            data.append(sequence)
    data = np.array(data)
    # transform into tensor
    data = torch.tensor(data, dtype=torch.float32)
    return data

class MotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        
        # Select the last time step's output
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# keypoints_file_path = "normalized_keypoints_data.json"
# video_label = "test"
# person_id = 0

# # get path of text file containing preprocessed data (specified by video and person id)
# # data_path = preprocess_data(keypoints_file_path, video_label, person_id)
# data_path = "./data/motion_keypoints/test/keypoints_seq_0.txt"

# window_size = 3
# data = load_data(data_path, window_size)
# frame_num = 2 #not less than window_size - 1
# data = data[frame_num - (window_size - 1)] #get one sequence only
# print(data)




# # Define the model and specify hyperparameters
# input_size = 36
# hidden_size = 64
# num_layers = 1
# output_size = 1

# model = MotionLSTM(input_size, hidden_size, num_layers, output_size)
# model.to(device)

# print(model)

# model.eval()  # Set the model in evaluation mode

# with torch.no_grad():
#     outputs = model(data)

# print(outputs)