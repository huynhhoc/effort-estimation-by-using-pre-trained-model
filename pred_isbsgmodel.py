import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from isbsg.pretrainedmodel import load_pretrained_model
from utils.utils import savetofile
# Load the pre-trained model
model = load_pretrained_model()
df = pd.read_csv('./dataset/isbsgsample.csv')
features = ['EI', 'EO', 'EQ', 'ILF', 'EIF', 'IndustrySector']
X = df[features]
y = df['SWE']
#encode the categorical variable 'IndustrySector' using the LabelEncoder:
le = LabelEncoder()
X.loc[:, 'IndustrySector'] = le.fit_transform(X['IndustrySector'])
X = X.values
y = y.values.reshape(-1, 1)

# Convert the input data to a PyTorch tensor
Xtest = torch.tensor(X, dtype=torch.float32)
# Make predictions on the test data
with torch.no_grad():
    ypred = model(Xtest)
# Convert the predicted values to a NumPy array
savetofile(Xtest,y, ypred, features, "./outputs/outputISBSG.csv")
print("---done---")
