'''
1. Design model (input, output, forward pass)
2. Cosntruct loss and optimizer
3. Training loop 
- forward pass: compute predictions 
- backward pass : gradients 
- Update weights 

'''
import torch
import torch.nn as nn

# f = w * x (predictor function)
# f = 2 * x (create artficial training data for func to est.)

# Comparison to prev. tutorial
# These are [4,] tensors, nn.Linear needs a 2D tensor 
# X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features
# Input dim, output dim (Only linear regression)
# model = nn.Linear(n_features, n_features)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        # Initialize default nn.module constructor and associated
        # attributes
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, X):
        return self.lin(X)

model = LinearRegression(input_size, output_size)

print (f"Prediction before training {model(X_test).item()}")

num_iters = 100
lr = 0.1
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(0, num_iters):
    # Prediction = forward pass 
    Y_pred = model(X)
    # Loss
    l = loss(Y, Y_pred)
    # Compute gradients using autograd
    l.backward() # dJ / dW
    optimizer.step()
    # Zero grad
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print (f"Epoch {epoch+1}: w= {w[0].item():.3f}, loss = {l:.8f}")

print (f"Prediction after training {model(X_test).item()}")



