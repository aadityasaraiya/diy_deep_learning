'''
epoch = 1 forward and backward pass of all training samples
batch_size = no. of training samples in one forward and backward pass

no. iters = no. of passes, each pass using batch size no. of samples 

eg: 100 samples, batch_size = 20 -> 100 / 20 iters = 5 iters per epoch

'''

import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math 

class WineDataset(Dataset):
    def __init__(self):
        # data loading 
        xy = np.loadtxt('wine.csv', delimiter=",", 
                        dtype=np.float32, skiprows=1)
        self.X = torch.from_numpy(xy[:, 1:])
        # n_samples, 1 
        self.Y = torch.from_numpy(xy[:, [0]])
        self.num_samples = xy.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])
        # dataset[0]
    
    def __len__(self):
        return self.num_samples

dataset = WineDataset()

batch_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

num_epochs = 2
total_samples = len(dataset)
num_iters = math.ceil(total_samples / batch_size)

print (total_samples, num_iters)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward, backward, update
        print (f"epoch {epoch + 1 / num_epochs}, step {i + 1}/ {num_iters}, inputs {inputs.shape}")
