from tinygrad.tensor import Tensor

x0 = Tensor([1])
x1 = Tensor([1]).log()
# x1.realize()
print(x0.cat(x1).numpy())