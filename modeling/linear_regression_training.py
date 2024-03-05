import numpy as np
from numpy.typing import NDArray


def get_derivative(model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], 
                   N: int, X: NDArray[np.float64], desired_weight: int) -> float:
    # Dot product of error with particular column input corresponding to the weight (assuming num_cols = num_weights= num_features)
    return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

def get_model_prediction(X: NDArray[np.float64], 
                         weights: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.squeeze(np.matmul(X, weights))

learning_rate = 0.01

def train_model(
    X: NDArray[np.float64], 
    Y: NDArray[np.float64], 
    num_iterations: int, 
    initial_weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    W = initial_weights
    N = X.shape[0]
    for i in range(0, num_iterations):
        # Perform forward pass (2 x 1)
        Y_hat = get_model_prediction(X, W) 
        # Update each weight
        for idx in range(0, len(W)):
            deriv = get_derivative(model_prediction=Y_hat, ground_truth=Y, N=N, X=X, desired_weight=idx)
            W[idx]-= deriv * learning_rate
    return np.round(W, 5)


# Input:
X = np.array([[1, 2, 3], [1, 1, 1]])
Y = np.array([6, 3])
num_iterations = 10
initial_weights = np.array([0.2, 0.1, 0.6])

output_weights = train_model(X, Y, num_iterations, initial_weights)
print (output_weights)

# Expected Output:
# [0.50678, 0.59057, 1.27435]
