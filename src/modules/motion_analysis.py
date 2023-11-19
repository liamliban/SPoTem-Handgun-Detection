import numpy as np
import torch
import torch.nn as nn
import random, os

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

# get one sequence of motion keypoint sets based on window size
def get_one_sequence(file_path, frame_num, window_size):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        if len(lines) <= frame_num:
            raise ("frame num is not in the text file!")

        # if window_size > len(lines):
        #     return None
        sequence = []
        for i in range(frame_num - (window_size - 1), frame_num + 1):
            if i < 0:
                string = '999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999'
                line = string.strip().split(',')
            else:
                line = lines[i].strip().split(',')
            sequence.append([float(val) for val in line])
        # data.append(sequence)
        data=sequence
    data = np.array(data)
    # transform into tensor
    data = torch.tensor(data, dtype=torch.float32)
    return data

class MotionLSTM(nn.Module):
    def __init__(self, hidden_size=20, lstm_layers=1):
        super(MotionLSTM, self).__init__()
        self.input_size = 36
        self.hidden_size = hidden_size
        self.num_layers = lstm_layers
        self.output_size = hidden_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        
        # Select the last time step's output
        out = out[:, -1, :]
        # out = self.fc(out)
        return out
