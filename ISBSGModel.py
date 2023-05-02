
"""
Written by Huynh Thai Hoc
April 26, 2023

Feel free to use and modify this program as you see fit.
"""

import torch.nn as nn
class ISBSGModel(nn.Module):
    def __init__(self,input_size):
        super(ISBSGModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x