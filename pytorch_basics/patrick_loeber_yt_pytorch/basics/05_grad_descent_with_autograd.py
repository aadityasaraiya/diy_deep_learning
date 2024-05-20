import torch

# f = w * x (predictor function)
# f = 2 * x (create artficial training data for func to est.)
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Model prediction 
def forward(X):
    return w * X 
# Loss 
def loss(Y, Y_pred):
    return ((Y_pred - Y)**2).mean()
# Calculate gradient 
# MSE = J=  1/ N * (w*x - y)**2
# dj/dW = 1/N * 2* (wx -y) * x
def gradient(X, Y, Y_pred):
    return np.dot(2 * X, Y_pred - Y).mean()

print (f"Prediction before training {forward(5)}")

num_iters = 100
learning_rate = 0.01

for epoch in range(0, num_iters):
    # Prediction = forward pass 
    Y_pred = forward(X)
    # Loss
    l = loss(Y, Y_pred)
    # Compute gradients using autograd
    l.backward() # dJ / dW

    # Ensure that we don't update gradients when we update w 
    with torch.no_grad():
        # Weights 
        w-= learning_rate * w.grad
    
    # Zero grad
    w.grad.zero_()

    if epoch % 1 == 0:
        print (f"Epoch {epoch+1}: w= {w:.3f}, loss = {l:.8f}")

print (f"Prediction after training {forward(5)}")



