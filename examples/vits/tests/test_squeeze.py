import torch
import numpy as np

from examples.vits.tiny.custom_ops import squeeze_tinygrad
from tinygrad.tensor import Tensor


def test_squeeze():


  # Create a dummy tensor with some dimensions of size 1
  dummy = np.ones((1, 3, 1, 4, 1, 5, 1), dtype=np.float32)

  # Create a tinygrad tensor from the dummy
  tinygrad_tensor = Tensor(dummy)

  # Squeeze all dimensions of size 1
  tinygrad_squeezed = squeeze_tinygrad(tinygrad_tensor)

  # Create a PyTorch tensor from the dummy
  torch_tensor = torch.from_numpy(dummy)

  # Squeeze all dimensions of size 1 in PyTorch
  torch_squeezed = torch_tensor.squeeze()

  # Compare the squeezed shapes
  assert tinygrad_squeezed.shape == torch_squeezed.shape, "Shape mismatch between tinygrad and torch squeeze."

  # Test squeezing a specific dimension
  dim_to_squeeze = 2

  # Squeeze the specific dimension in tinygrad
  tinygrad_squeezed_dim = squeeze_tinygrad(tinygrad_tensor, axis=dim_to_squeeze)

  # Squeeze the specific dimension in PyTorch
  torch_squeezed_dim = torch_tensor.squeeze(dim=dim_to_squeeze)

  # Compare the squeezed shapes
  assert tinygrad_squeezed_dim.shape == torch_squeezed_dim.shape, "Shape mismatch between tinygrad and torch squeeze at specific dimension."


if __name__ == '__main__':
    test_squeeze()