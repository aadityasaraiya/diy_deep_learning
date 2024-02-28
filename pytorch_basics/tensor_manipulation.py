"""
Summary:
- Sum along a specified axis in a tensor.
- Squeeze and unsqueeze operations on a tensor.
- Reshape a tensor into a desired shape.
- Calculate the average across rows of a tensor.
- Concatenate two tensors along a specified dimension.
- Compute the mean squared error loss between two tensors.

Functions:
1. torch_sum(a: TensorType[float], axis: int) -> TensorType[float]:
    - Calculate the sum of elements along the specified axis.

2. tensor_squeeze_unsqueeze(a: TensorType[float]) -> TensorType[float]:
    - Demonstrate squeeze and unsqueeze operations on a tensor.

3. tensor_reshape(to_reshape: TensorType[float]) -> TensorType[float]:
    - Reshape a tensor into an M x N // 2 x tensor.

4. tensor_avg(to_avg: TensorType[float]) -> TensorType[float]:
    - Calculate the average across rows of a tensor.

5. torch_concat(cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
    - Concatenate two tensors along the second dimension.

6. get_loss(prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
    - Compute the mean squared error loss between prediction and target tensors.
"""

import torch 
from torchtyping import TensorType

def tensor_reshape(to_reshape: TensorType[float]) -> TensorType[float]:
    """
    Reshape an input tensor to have half the number of columns.

    Parameters:
    - to_reshape (TensorType[float]): Input tensor to be reshaped.

    Returns:
    - TensorType[float]: Reshaped tensor.
    """
    return torch.reshape(to_reshape, ((to_reshape.shape[0] * to_reshape.shape[1]) // 2, 2))


def tensor_avg(to_avg: TensorType[float]) -> TensorType[float]:
    """
    Calculate the average across rows of a given tensor.

    Parameters:
    - to_avg (TensorType[float]): Input tensor for which the row-wise average is calculated.

    Returns:
    - TensorType[float]: Row-wise average tensor.
    """
    return torch.mean(to_avg, dim=0)


def torch_concat(cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
    """
    Concatenate two tensors along the specified dimension.

    Parameters:
    - cat_one (TensorType[float]): First tensor to be concatenated.
    - cat_two (TensorType[float]): Second tensor to be concatenated.

    Returns:
    - TensorType[float]: Concatenated tensor.
    """
    return torch.cat((cat_one, cat_two), dim=1)


def get_loss(prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
    """
    Compute the mean squared error loss between two tensors.

    Parameters:
    - prediction (TensorType[float]): Predicted values tensor.
    - target (TensorType[float]): Target values tensor.

    Returns:
    - TensorType[float]: Mean squared error loss.
    """
    return torch.nn.functional.mse_loss(prediction, target)


# Example: Reshape a tensor
to_reshape = torch.tensor([
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
])
after_reshape_expected = torch.tensor([
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0]
])

assert torch.equal(after_reshape_expected, tensor_reshape(to_reshape)), "Reshape not as expected"

# Example: Average of a tensor
to_avg = torch.tensor([
    [0.8088, 1.2614, -1.4371],
    [-0.0056, -0.2050, -0.7201]
])

expected_avg = torch.tensor([0.4016, 0.5282, -1.0786])

assert torch.equal(tensor_avg(to_avg), expected_avg), "Average not as expected"

# Example: Concatenate tensors
cat_one = torch.tensor([
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
])

cat_two = torch.tensor([
    [1.0, 1.0],
    [1.0, 1.0]
])

expected_post_concat = torch.tensor([
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0]
])

assert torch.equal(torch_concat(cat_one, cat_two), expected_post_concat), "Concatenation not as expected"

# Example: Compute loss
prediction = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
expected_loss = torch.tensor(0.6)

assert torch.equal(get_loss(prediction, target), expected_loss), "Loss not as expected"