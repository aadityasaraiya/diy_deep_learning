'''
1) Design model (input, output)
2) Construct loss and optimizer
3) Training loop 
- Forward pass: compute predictions and loss 
- Backward pass: gradients 
- Update weights 
'''

import torch 
import torch.nn as nn
import numpy as np 
from sklearn import datasets
from matplotlib import pyplot as plt
# import matplotlib.pylot as plt

# S0: Prepare data 
# X_numpy : 100x 1, Y_numpy : 100,
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, 
                                       noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)
n_samples, n_features = X.shape

# S1: Design model 
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, X):
        return self.lin(X)

model = LinearRegression(input_dim=n_features, output_dim=n_features)

# S2: Define loss and optimizer
lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# S3: Training loop
num_epochs = 100

for epoch in range(0, num_epochs):
    # Perform prediction
    Y_pred = model(X)
    # Compute loss
    loss = criterion(Y_pred, Y)
    # Calculate gradients 
    loss.backward()
    # Update weights 
    optimizer.step()
    # Zero out previously calculated gradients 
    optimizer.zero_grad()

# Detach() -> original tensor but without requiring grad compute
predicted = model(X).detach()

plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()



