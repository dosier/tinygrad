import numpy as np
import torch

from examples.vits.tiny.custom_ops import masked_fill_tinygrad
from tinygrad.tensor import Tensor

# Test
def test_masked_fill():
    # Prepare data
    data = np.array([1, 2, 3, 4], dtype=np.float32)
    mask_data = np.array([True, False, True, False])
    value = -1

    # PyTorch computation
    torch_tensor = torch.tensor(data)
    torch_mask = torch.tensor(mask_data)
    torch_result = torch_tensor.masked_fill(torch_mask, value).numpy()

    # tinygrad computation
    tinygrad_tensor = Tensor(data)
    tinygrad_mask = Tensor(mask_data.astype(np.float32))  # Convert mask to float
    tinygrad_result = masked_fill_tinygrad(tinygrad_tensor, tinygrad_mask, value).numpy()

    # Assert that the two results are almost equal (within tolerance due to potential minor differences in computation)
    assert np.allclose(torch_result, tinygrad_result, atol=1e-6), f"Expected {torch_result}, but got {tinygrad_result}"

if __name__ == '__main__':

  # Run the test
  test_masked_fill()