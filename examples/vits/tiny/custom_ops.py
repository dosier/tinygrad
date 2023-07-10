import numpy
import numpy as np

from tinygrad.tensor import Tensor

def norm_except_dim(v, dim):
  if dim == -1:
    return np.linalg.norm(v)
  elif dim == 0:
    output_shape = [1] * v.ndim
    output_shape[0] = v.shape[0]
    return np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).reshape(output_shape)
  elif dim == v.ndim - 1:
    output_shape = [1] * v.ndim
    output_shape[-1] = v.shape[-1]
    return np.linalg.norm(v.reshape(-1, v.shape[-1]), axis=0).reshape(output_shape)
  else:
    transposed_v = np.transpose(v, (dim,) + tuple(i for i in range(v.ndim) if i != dim))
    return np.transpose(norm_except_dim(transposed_v, 0), (dim,) + tuple(i for i in range(v.ndim) if i != dim))

def weight_norm_tinygrad(v, g, dim):
  v, g = v.numpy(), g.numpy()
  v_norm = norm_except_dim(v, dim)
  w = v * (g / v_norm)
  return Tensor(w)

def masked_fill_tinygrad(tensor, mask, value):
  return tensor * (1 - mask) + value * mask

def split_tinygrad(tensor, split_sizes, dim=0):
  # if split_sizes is an integer, convert it to a tuple of size split_sizes elements
  if isinstance(split_sizes, int):
    split_sizes = (split_sizes,) * (tensor.shape[dim] // split_sizes)

  assert sum(split_sizes) == tensor.shape[
    dim], "Sum of split_sizes must equal the dimension size of tensor along the given dimension."

  start = 0
  slices = []
  for size in split_sizes:
    slice_range = [(start, start + size) if j == dim else None for j in range(len(tensor.shape))]
    slices.append(slice_range)
    start += size
  return [tensor.slice(s) for s in slices]

def bitwise_and_tinygrad(tensor1, tensor2):
  assert tensor1.shape == tensor2.shape, "Tensors must have the same shape."
  a, b = tensor1.astype(numpy.uint8).astype(numpy.uint8), tensor2.numpy().astype(numpy.uint8)
  return Tensor(a & b, device=tensor1.device, dtype=tensor1.dtype)


def squeeze_tinygrad(tensor, axis=None):
  if axis is None:
    new_shape = [dim for dim in tensor.shape if dim != 1]
  else:
    assert tensor.shape[axis] == 1, "Cannot squeeze dim {} with size {}".format(axis, tensor.shape[axis])
    new_shape = list(tensor.shape)
    new_shape.pop(axis)
  return tensor.reshape(*new_shape)