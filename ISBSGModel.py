
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
        # Apply feature mapping functions to x to adapt to the model's input size
        x = self.feature_mapping(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def feature_mapping(self, x):
        # Implement feature mapping logic here
        # This function should map x from the new dataset to the model's input size
        # Example: If the new dataset has a different feature order or scaling, adjust it here
        
        # Assuming the input features are in the order EI, EO, EQ, EIF, ILF
        # and the new dataset has them in a different order
        # You can reorder them to match the model's input order
        x_reordered = x[:, [3, 0, 1, 2, 4]]  # New order: EIF, EI, EO, EQ, ILF
        # You can also apply any necessary scaling or transformations here
        return x_reordered
