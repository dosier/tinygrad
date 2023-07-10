import numpy as np
import torch

from tinygrad.tensor import Tensor

np.random.seed(0)  # Make sure both tests use the same random data
data = np.random.rand(2, 1, 10).astype(np.float32)


def _tinygrad_weight_norm(v, g, dim):
  # Calculate L2 norm of each row in v
  v_norm = np.linalg.norm(v.numpy(), ord=2, axis=dim, keepdims=True)
  # Normalize each row in v by dividing it by its L2 norm
  v_normalized = v.numpy() / v_norm
  # Rescale the normalized v by the learnable parameter g
  scaled_weights = g.numpy() * v_normalized
  return Tensor(scaled_weights)

# Tinygrad version
g_tinygrad = Tensor(data.copy())
v_tinygrad = Tensor(data.copy())
weight_tinygrad = _tinygrad_weight_norm(v_tinygrad, g_tinygrad, 0)
print("Tinygrad weights: ", weight_tinygrad.numpy())

# PyTorch version
g_torch = torch.tensor(data.copy(), requires_grad=True)
v_torch = torch.tensor(data.copy(), requires_grad=True)
v_norm_torch = torch.nn.functional.normalize(v_torch, p=2, dim=0)
weight_torch = g_torch * v_norm_torch
print("PyTorch weights: ", weight_torch.detach().numpy())

# Check if they're close
np.testing.assert_allclose(weight_tinygrad.numpy(), weight_torch.detach().numpy(), atol=1e-7)
