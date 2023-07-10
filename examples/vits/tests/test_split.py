import torch

from examples.vits.tiny.custom_ops import split_tinygrad
from tinygrad.tensor import Tensor
import numpy as np

def test_split():
    # Prepare data
    data = np.random.randn(8, 3).astype(np.float32)
    split_size = 4

    # PyTorch computation
    torch_tensor = torch.tensor(data)
    torch_splits = torch_tensor.split(split_size, dim=0)
    torch_splits = [t.numpy() for t in torch_splits]

    # tinygrad computation
    tinygrad_tensor = Tensor(data)
    tinygrad_splits = split_tinygrad(tinygrad_tensor, split_size, dim=0)
    tinygrad_splits = [t.numpy() for t in tinygrad_splits]

    # Assert that the two results are the same
    for torch_split, tinygrad_split in zip(torch_splits, tinygrad_splits):
        assert np.allclose(torch_split, tinygrad_split, atol=1e-6), f"Expected {torch_split}, but got {tinygrad_split}"

# Run the test
if __name__ == '__main__':
    test_split()
