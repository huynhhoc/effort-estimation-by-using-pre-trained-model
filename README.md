# EFFORT ESTIMATION BY USING PRE-TRAINED MODEL
This repository provides a pre-trained model to estimate software effort estimation. The pre-trained model is trained based on the ISBSG dataset (Release R1, 2020).

# ISBSG Repository

The International Software Benchmarking Standards Group (ISBSG) Repository is a database of software project data that can be used for benchmarking, estimating, and improving software development processes. The repository contains a collection of historical data from thousands of completed software projects, including information about project size, effort, duration, defects, and other relevant metrics.

The ISBSG Repository is maintained by the ISBSG, a not-for-profit organization that aims to improve software development practices by providing reliable data and benchmarks. The organization has members from around the world, including software development organizations, government agencies, and academic institutions.

The ISBSG Repository is a valuable resource for organizations that want to improve their software development processes, estimate the costs and resources required for new projects, and compare their performance with industry benchmarks. The data in the repository can also be used by researchers and academics to study software development trends and practices.

The ISBSG dataset (Release 2020) contains 9,592 finished software projects. There are a total of 251 documented attributes. These are separated into a variety of categories, such as Summary Work Effort (SWE), total effort in hours recorded against the project; AFP, the adjusted functional size of the project at the final count; VAF, the adjustment to the function points that take into account various technical and quality characteristic; functional point variables (EI, EO, EQ, EIF, ELF), PDR, and other categorical variables such as Industry Sector, Relative Size, etc.

# PRE-TRAINED MODEL

## Pre-trained model
The pre-trained model has been trained based on ISBSG repository (release R1, 2020). The ISBSG dataset (2020) is filtered using four criteria. Firstly, projects with insufficient data or fewer credible data are eliminated. This action reduces the scope of the dataset to 8,619 projects. Secondly, the counting approach focuses only on IFPUG’s FPA, and the number of selected projects decreases to 6,365. Thirdly, all uncounted attributes are excluded, leaving 1,515 projects. Lastly, remove outliers for PDR values in the dataset. EI, EO, EQ, EIF, ILF, and Industry Sector are used as input features, and SWE is the output.

## ISBSGModel:

```
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

```

## Install pre-trained model

Navigate to the ``dist`` directory and run the following command:

```
    pip install isbsg-0.1.0.tar.gz

```
This should install isbsg package into the virtual environment. You can then use it in your Python projects by importing it like this:

```
    from isbsg.pretrainedmodel import load_pretrained_model

```
You can uninstall the package by using the pip uninstall command in your terminal or command prompt:

```
 pip uninstall isbsg

```
# How to load pre-trained model

Transfer learning based on a pre-trained model involves several steps as follows:
* Step 1: Load the pre-trained model: This model is trained on an ISBSG dataset.
* Step 2: Freeze all layers except the last one: all layers, except for the last one, are frozen. Freezing involves setting the requires_grad attribute of each parameter to False, which preserves the learned features from the pre-trained model. This step enables fine-tuning of only the last layer for the new task.
* Step 3: Create a new optimiser: a new optimiser is created specifically for the last layer of the model, which was set to require gradients in the previous step. Adam optimiser is chosen as the same as the PyDL model. 
* Step 4: Continue training the model: Train the model on a new dataset using the standard PyTorch training loop. In each iteration, the input is forwarded through the model, the loss is computed, the gradients are computed, and the weights are updated using the optimiser. This process is repeated for a certain number of epochs or until the model converges. 

Here is an example of step 1, and 2:

```
from isbsg.pretrainedmodel import load_pretrained_model

# Step 1: Load pre-trained model
pretrained_model = load_pretrained_model()
# Step 2: Freeze all layers except the last one
for param in pretrained_model.parameters():
    param.requires_grad = False
# Replace the last layer with a new input layer
pretrained_model.fc1 = nn.Linear(new_input_size, pretrained_model.fc1.out_features)

```

# SAMPLE

Desharnais and Albrecht are selected for illustrating because their efforts have the same metrics as the effort in the ISBSG. The trained model obtained from Desharnais and Albrecht are: ``./weights/desharnais.pth`` and ``./weights/albrecht.pth``.

To test the performance, we can use:

```
python pred_desharnai.py, or python pred_albrecht.py

```


