import torch

from examples.vits.tiny.custom_ops import bitwise_and_tinygrad
from tinygrad.tensor import Tensor
import numpy as np

def test_bitwise_and():
    # Prepare data
    data1 = np.array([1, 2, 3], dtype=np.uint8)
    data2 = np.array([2, 3, 4], dtype=np.uint8)

    # PyTorch computation
    torch_tensor1 = torch.tensor(data1)
    torch_tensor2 = torch.tensor(data2)
    torch_result = (torch_tensor1 & torch_tensor2).numpy()

    # tinygrad computation
    tinygrad_tensor1 = Tensor(data1)
    tinygrad_tensor2 = Tensor(data2)
    tinygrad_result = bitwise_and_tinygrad(tinygrad_tensor1, tinygrad_tensor2).numpy()

    # Assert that the results are equal
    assert np.allclose(torch_result, tinygrad_result, atol=1e-6), f"Expected {torch_result}, but got {tinygrad_result}"


if __name__ == '__main__':
  # Run the test
  test_bitwise_and()