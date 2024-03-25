import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Define the architecture here
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        out1 = self.dropout(self.relu(self.layer1(torch.flatten(images, start_dim=1))))
        out2 = self.sigmoid(self.layer2(out1))

        return torch.round(out2, decimals=4)

# After initialization, if this network is called, it accepts an N x 28 x 28 tensor and performs forward pass 
# Output is a N x 10 tensor, which represents the classes from 0 - 9  