# Load the dataset
import pandas as pd
import torch
from ISBSGModel import *
from utils.utils import savetofile
#Load dataset
df = pd.read_csv('./dataset/albrecht.csv')
features = ['Input', 'Output', 'Inquiry','File','FPAdj', 'RawFPCount', 'AdjFP']
response_variable = 'Effort'
Xtest = df[features].values
yreal = df[response_variable].values.reshape(-1, 1)

# Define the path to the .pth file
path_to_model = "./weights/albrecht.pth"

# Load the model
model = ISBSGModel(input_size=len(features))
model.load_state_dict(torch.load(path_to_model))

# Convert the input data to a PyTorch tensor
Xtest = torch.tensor(Xtest, dtype=torch.float32)
# Make predictions on the test data
with torch.no_grad():
    ypred = model(Xtest)

# Convert the predicted values to a NumPy array
savetofile(Xtest,yreal, ypred, features, "./outputs/outputAlbrecht.csv")
print("---done---")