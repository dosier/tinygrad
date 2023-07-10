import numpy as np
import torch
import torch.nn.functional as F

from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor

unnormalized_derivatives = np.array([[1, 2, 3], [4, 5, 6]])

# NumPy padding
numpy_padded = np.pad(unnormalized_derivatives, pad_width=((0, 0), (1,0)), mode='constant', constant_values=0)
print(numpy_padded)

# PyTorch padding
torch_tensor = torch.tensor(unnormalized_derivatives)
torch_padded = F.pad(torch_tensor, pad=(1, 0), mode='constant', value=0)
print(torch_padded)

print(Tensor.arange(10, dtype=dtypes.int32, device="CPU").numpy())

print(np.promote_types(dtypes.int32.np,dtypes.int32.np))