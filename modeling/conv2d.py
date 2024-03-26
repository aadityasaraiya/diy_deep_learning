import numpy as np
import torch

# Fix seed for NumPy random number generator
np.random.seed(42)
# Fix seed for PyTorch random number generator
torch.manual_seed(42)

def conv2d(image, kernel, stride, padding):
    # Step 1: Pad input image 
    padded_img = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    # Step 2: find output image dims 
    H = ((padded_img.shape[0] - kernel.shape[0]) // stride[0]) + 1
    W = ((padded_img.shape[1] - kernel.shape[1]) // stride[1]) + 1

    output_img = np.zeros((H, W))

    for i in range(0, H):
        for j in range (0, W):
            window = padded_img[i * stride[0] : i * stride[0] + kernel.shape[0], j * stride[1] : j * stride[1] + kernel.shape[1]]
            output_img[i, j] = np.sum(kernel * window)
    return output_img

# Define the input images, kernels, strides, and padding
image1 = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

image2 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Define the kernels (filters)
kernel1 = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])

kernel2 = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])

stride = (2, 2)
padding = 2  # Example padding value

# Convert numpy arrays to PyTorch tensors
torch_image1 = torch.tensor(image1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
torch_image2 = torch.tensor(image2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
torch_kernel1 = torch.tensor(kernel1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add input and output channel dimensions
torch_kernel2 = torch.tensor(kernel2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Apply convolution using PyTorch with explicit stride and padding
torch_output1 = torch.nn.functional.conv2d(torch_image1, torch_kernel1, stride=stride, padding=padding)
torch_output2 = torch.nn.functional.conv2d(torch_image2, torch_kernel2, stride=stride, padding=padding)

# Convert PyTorch tensor outputs to numpy arrays
expected_output1 = torch_output1.squeeze().numpy()
expected_output2 = torch_output2.squeeze().numpy()

# Call custom conv2d function with explicit stride and padding
custom_output1 = conv2d(image1, kernel1, stride, padding)
custom_output2 = conv2d(image2, kernel2, stride, padding)

# Print outputs
print("PyTorch Output 1:")
print(expected_output1)
print("\nCustom Function Output 1:")
print(custom_output1)
print("\nPyTorch Output 2:")
print(expected_output2)
print("\nCustom Function Output 2:")
print(custom_output2)

# Assert that the outputs match
assert np.array_equal(custom_output1, expected_output1), "Test 1 failed: Outputs do not match"
assert np.array_equal(custom_output2, expected_output2), "Test 2 failed: Outputs do not match"

print("Custom conv2d outputs match PyTorch outputs.")
