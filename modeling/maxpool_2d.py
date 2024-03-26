import numpy as np
import torch

def maxpool2d(image, kernel_size, stride, padding):
    # Step 1: Apply padding to image 
    padded_img = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    # Step 2: Find dimensions of the output image from the padded image 
    H = ((padded_img.shape[0] - kernel_size[0]) // stride[0]) + 1 
    W = ((padded_img.shape[1] - kernel_size[1]) // stride[1]) + 1 
    output_img = np.zeros((H, W))

    # Step 3: Iterate through entire dims of output image 
    for i in range (0, H):
        for j in range(0, W):
            # Step 4: Extract window at each step. Window is defined by 4 numbers, x, x+ delta, y, y+ delta. x and y itself 
            # jump with the stride. delta= kernel size 
            window = padded_img[i * stride[0] : i * stride[0] + kernel_size[0], j * stride[1] : j * stride[1] + kernel_size[1]]
            # Step 5: Apply filter, in this case, its the max operator 
            output_img[i, j] = np.max(window)
    return output_img


# Define the input images, pool sizes, strides, and padding
image1 = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

image2 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

kernel_size = (3, 3)
stride = (2, 2)
padding = 1

# Convert numpy arrays to PyTorch tensors
torch_image1 = torch.tensor(image1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
torch_image2 = torch.tensor(image2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Apply max pooling using PyTorch with explicit stride and padding
torch_output1 = torch.nn.functional.max_pool2d(torch_image1, kernel_size=kernel_size, stride=stride, padding=padding)
torch_output2 = torch.nn.functional.max_pool2d(torch_image2, kernel_size=kernel_size, stride=stride, padding=padding)

# Convert PyTorch tensor outputs to numpy arrays
expected_output1 = torch_output1.squeeze().numpy()
expected_output2 = torch_output2.squeeze().numpy()

# Call custom maxpool2d function with explicit stride and padding
custom_output1 = maxpool2d(image1, kernel_size, stride, padding)
custom_output2 = maxpool2d(image2, kernel_size, stride, padding)

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

print("Custom maxpool2d outputs match PyTorch outputs.")
