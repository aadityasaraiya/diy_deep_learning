'''
1. Design model (input, output, forward)
2. Construct loss and optimizer 
3. Training loop 
- forward pass: compute preds and loss 
- backward pass: gradients 
- update weights 
'''

import torch 
import torch.nn as nn
import numpy as np 

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

# 0) prepare data 
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target
n_samples, n_features = X.shape 
print (n_samples, n_features)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# Scale features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))
# To make it into a column vector
Y_train = Y_train.view(Y_train.shape[0], 1)
Y_test = Y_test.view(Y_test.shape[0], 1)

# 1) Model 
# f = wx + b, sigmoid at end 

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        # binary classifier has 1 class label at end
        self.linear = nn.Linear(n_input_features, 1)
    def forward(self, X):
        Y_pred = torch.sigmoid(self.linear(X))
        return Y_pred
model = LogisticRegression(n_features)

# 2) Loss and optimizer 
lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#3) Training loop 
n_epochs = 100
for epoch in range(0, n_epochs):
    # Perform forward 
    Y_pred = model(X_train)
    # Compute Loss 
    loss = criterion(Y_pred, Y_train)
    # Perform backward 
    loss.backward()
    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch - 1) % 10 == 0:
        print (f"Epoch {epoch-1}, Loss {loss.item():.4f}, ")

# Evaluation 
with torch.no_grad():
    Y_pred = model(X_test)
    Y_pred_cls = Y_pred.round()
    acc = Y_pred_cls.eq(Y_test).sum() / float(Y_test.shape[0])

    print (f"Accuracy {acc}")


