import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self):
        super(MotionLSTM, self).__init__()
        self.input_size = 36
        self.hidden_size = 20
        self.num_layers = 1
        self.output_size = 20
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
