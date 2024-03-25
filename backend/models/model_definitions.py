import torch.nn as nn



import sys
import os # adding filepaths list from data_processing directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data_processing import data_config
filepaths = data_config.filepaths
names = data_config.names


class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.layer_1 = nn.Linear(384, 64)  
        self.layer_2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, len(names))  
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.output_layer(x)
        return x
