import torch
from tqdm import tqdm
import json

from src.model import handpose_model
import numpy as np
import random

# Set a random seed for reproducibility
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)

torch.backends.cudnn.deterministic=True

model = handpose_model()

size = {}
for i in tqdm(range(10, 1000)):
    data = torch.randn(1, 3, i, i)
    if torch.cuda.is_available():
        data = data.cuda()
    size[i] = model(data).size(2)

with open('hand_model_output_size.json') as f:
    json.dump(size, f)
