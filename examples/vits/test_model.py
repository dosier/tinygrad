import time
import unittest

import numpy
import torch

from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor
from examples.vits.tiny import attention as tiny_attention, models as tiny_models, modules as tiny_modules
from examples.vits.tiny.modules import fused_add_tanh_sigmoid_multiply as tiny_fused_add_tanh_sigmoid_multiply
from examples.vits.tiny.inference import load_model as tiny_load_model
from examples.vits.tiny.inference import get_text as tiny_get_text
from examples.vits.torchy import attentions as torch_attention, models as torch_models, modules as torch_modules
from examples.vits.torchy.commons import fused_add_tanh_sigmoid_multiply as torch_fused_add_tanh_sigmoid_multiply
from examples.vits.torchy.run import load_model as torch_load_model
from examples.vits.torchy.run import get_text as torch_get_text

class TestModelLoad(unittest.TestCase):
  def setUp(self):
    Tensor.manual_seed(1337)  # Deterministic
    numpy.random.seed(1337)
    torch.manual_seed(1337)
    self.tiny_model, self.tiny_hps = tiny_load_model()
    self.torch_model, self.torch_hps = torch_load_model()
    tiny_stn_tst = tiny_get_text("Tinygrad is Awesome!", self.tiny_hps)
    self.tiny_x, self.tiny_x_length = tiny_stn_tst.unsqueeze(0), Tensor([tiny_stn_tst.shape[0]], dtype=dtypes.int64)
    torch_stn_tst = torch_get_text("Tinygrad is Awesome!", self.torch_hps)
    self.torch_x, self.torch_x_length = torch_stn_tst.unsqueeze(0), torch.tensor([torch_stn_tst.shape[0]], dtype=torch.int64)

  def tiny_infer(self): return self.tiny_model.infer(self.tiny_x, self.tiny_x_length, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0].numpy()
  def torch_infer(self): return self.torch_model.infer(self.torch_x, self.torch_x_length, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0].numpy()

  @torch.no_grad()
  def test_speed(self):
    Tensor.training = False
    Tensor.no_grad = True
    warmup_rounds = 1
    infer_rounds = 5
    # Warm-up phase
    for _ in range(warmup_rounds):
      _ = self.tiny_infer()
      _ = self.torch_infer()

    # Measure the time taken by the tinygrad model
    start_time = time.time()
    for _ in range(infer_rounds):
      tiny_output = self.tiny_infer()
    tiny_time = (time.time() - start_time) / infer_rounds

    # Measure the time taken by the PyTorch model
    start_time = time.time()
    for _ in range(infer_rounds):
      torch_output = self.torch_infer()
    torch_time = (time.time() - start_time) / infer_rounds
    print("\n")
    print(f"Tinygrad model took {tiny_time} seconds per forward pass")
    print(f"PyTorch model took {torch_time} seconds per forward pass")


class TestFusion(unittest.TestCase):
  def setUp(self):
    Tensor.manual_seed(1337)  # Deterministic
    numpy.random.seed(1337)
    torch.manual_seed(1337)
    self.tiny_fusion = tiny_fused_add_tanh_sigmoid_multiply
    self.torch_fusion = torch_fused_add_tanh_sigmoid_multiply
  def test_fusion(self):
    a = numpy.random.randn(1, 384, 96).astype(numpy.float32)
    b = numpy.random.randn(1, 384, 96).astype(numpy.float32)
    n_channels = numpy.ones(1).astype(numpy.int32)
    tiny_res = self.tiny_fusion(Tensor(a.copy()), Tensor(b.copy()), Tensor(n_channels)).numpy()
    torch_res = self.torch_fusion(torch.from_numpy(a.copy()), torch.from_numpy(b.copy()), torch.from_numpy(n_channels)).numpy()
    self.assertTrue(numpy.allclose(tiny_res, torch_res))

class TestFlip(unittest.TestCase):

    def setUp(self):
      Tensor.manual_seed(1337)  # Deterministic
      numpy.random.seed(1337)
      torch.manual_seed(1337)
      self.tiny_flip = tiny_modules.Flip()
      self.torch_flip = torch_modules.Flip()

    @torch.no_grad()
    def test_forward(self):
      x = numpy.random.randn(2, 100).astype(numpy.float32)
      y = self.tiny_flip.forward(Tensor(x))[1].numpy()
      y2 = self.torch_flip.forward(torch.from_numpy(x))[1].numpy()
      self.assertTrue(numpy.allclose(y, y2))


class TestGenerator(unittest.TestCase):

  def setUp(self):
    Tensor.manual_seed(1337)  # Deterministic
    numpy.random.seed(1337)
    torch.manual_seed(1337)
    self.tiny_mha = tiny_models.Generator(192, '1', [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [8, 8, 2, 2], 512, [16, 16, 4, 4])
    self.torch_mha = torch_models.Generator(192, '1', [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [8, 8, 2, 2], 512, [16, 16, 4, 4])

  @torch.no_grad()
  def test_generator(self):
    x = numpy.random.rand(1,192, 95).astype(numpy.float32)
    g = numpy.random.rand(1, 65, 96).astype(numpy.float32)
    tiny_x_tensor, torch_x_tensor = Tensor(x.copy()), torch.from_numpy(x.copy())
    tiny_g_tensor, torch_g_tensor = Tensor(g.copy()), torch.from_numpy(g.copy())
    self.assertTrue(numpy.allclose(tiny_x_tensor.numpy(), torch_x_tensor.numpy()))
    tiny_res = self.tiny_mha.forward(tiny_x_tensor, None).numpy()
    torch_res = self.torch_mha.forward(torch_x_tensor, None).numpy()
    self.assertTrue(numpy.allclose(tiny_res, torch_res))

class TestMultiHeadAttention(unittest.TestCase):

  def setUp(self):
    Tensor.manual_seed(1337)  # Deterministic
    numpy.random.seed(1337)
    torch.manual_seed(1337)
    self.tiny_mha = tiny_attention.MultiHeadAttention(192, 192, 2)
    self.torch_mha = torch_attention.MultiHeadAttention(192, 192, 2)

  def test__absolute_position_to_relative_position(self):
    data = numpy.random.rand(1, 2, 33, 33).astype(numpy.float32)
    tiny_tensor, torch_tensor = Tensor(data.copy()), torch.from_numpy(data.copy())
    tiny_relative_weights = self.tiny_mha._absolute_position_to_relative_position(tiny_tensor).numpy()
    torch_relative_weights = self.torch_mha._absolute_position_to_relative_position(torch_tensor).numpy()
    self.assertTrue(numpy.allclose(tiny_relative_weights, torch_relative_weights))

  def test__relative_position_to_absolute_position(self):
    # x: [b, h, l, 2*l-1]
    # ret: [b, h, l, l]
    data = numpy.random.rand(1, 2, 33, 65).astype(numpy.float32)
    tiny_tensor, torch_tensor = Tensor(data.copy()), torch.from_numpy(data.copy())
    tiny_local_scores = self.tiny_mha._relative_position_to_absolute_position(tiny_tensor).numpy()
    torch_local_scores = self.torch_mha._relative_position_to_absolute_position(torch_tensor).numpy()
    self.assertTrue(numpy.allclose(tiny_local_scores, torch_local_scores))

  def test__matmul_with_relative_keys(self): # x: [b, h, l, d], y: [h or 1, m, d], re, : [b, h, l, m]
    q = numpy.random.rand(1, 2, 33, 96).astype(numpy.float32)
    d = numpy.random.rand(1, 65, 96).astype(numpy.float32)
    tiny_q_tensor, torch_q_tensor = Tensor(q.copy()), torch.from_numpy(q.copy())
    tiny_d_tensor, torch_d_tensor = Tensor(d.copy()), torch.from_numpy(d.copy())
    tiny_local_scores = self.tiny_mha._matmul_with_relative_keys(tiny_q_tensor, tiny_d_tensor).numpy()
    torch_local_scores = self.torch_mha._matmul_with_relative_keys(torch_q_tensor, torch_d_tensor).numpy()
    self.assertTrue(numpy.allclose(tiny_local_scores, torch_local_scores))

