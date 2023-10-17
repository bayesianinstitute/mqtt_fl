import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # Input Layer
        self.fc1 = nn.Linear(30, 300)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm1d(300)
        
        # Hidden Layers
        self.fc2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.batchnorm2 = nn.BatchNorm1d(100)
        
        self.fc3 = nn.Linear(100, 50)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.batchnorm3 = nn.BatchNorm1d(50)
        
        # Output Layer
        self.fc4 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.batchnorm3(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)
        
        return x


