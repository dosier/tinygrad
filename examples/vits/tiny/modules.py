import math

import tinygrad.nn as nn
from examples.vits.tiny.custom_ops import split_tinygrad
from examples.vits.tiny.transforms import piecewise_rational_quadratic_transform
from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor

LRELU_SLOPE = 0.1

def fused_add_tanh_sigmoid_multiply(input_a: Tensor, input_b: Tensor, n_channels: Tensor):
  n_channels_int, in_act = n_channels.numpy()[0], input_a + input_b
  t_act, s_act = in_act[:, :n_channels_int, :].tanh(), in_act[:, n_channels_int:, :].sigmoid()
  return t_act * s_act

def get_padding(kernel_size, dilation=1): return int((kernel_size*dilation - dilation)/2)

class LayerNorm(nn.LayerNorm):
  def __init__(self, channels, eps=1e-5): super().__init__(channels, eps, elementwise_affine=True)
  def forward(self, x: Tensor): return self.__call__(x.transpose(1, -1)).transpose(1, -1)

class WN:
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    assert (kernel_size % 2 == 1)
    self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.gin_channels, self.p_dropout = hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout
    self.in_layers, self.res_skip_layers = [], []
    if gin_channels != 0: self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding)
      self.in_layers.append(in_layer)
      res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
      res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, g=None, **kwargs):
    output, n_channels_tensor = Tensor.zeros_like(x), Tensor([self.hidden_channels], dtype=dtypes.int64)
    if g is not None: g = self.cond_layer(g)
    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
      else:
        g_l = Tensor.zeros_like(x_in)
      acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:, :self.hidden_channels, :]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:, self.hidden_channels:, :]
      else:
        output = output + res_skip_acts
    return output * x_mask

class ResBlock1:
  def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
    self.convs1 = [nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i], padding=get_padding(kernel_size, dilation[i])) for i in range(3)]
    self.convs2 = [nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)) for _ in range(3)]

  def forward(self, x: Tensor, x_mask=None):
    for c1, c2 in zip(self.convs1, self.convs2):
      xt = x.leakyrelu(LRELU_SLOPE)
      xt = c1(xt if x_mask is None else xt * x_mask).leakyrelu(LRELU_SLOPE)
      x = c2(xt if x_mask is None else xt * x_mask) + x
    return x if x_mask is None else x * x_mask

class ResBlock2:
  def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
    self.convs = [nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i], padding=get_padding(kernel_size, dilation[i])) for i in range(2)]

  def forward(self, x, x_mask=None):
    for c in self.convs:
      xt = x.leaky_relu(LRELU_SLOPE)
      xt = c(xt if x_mask is None else xt * x_mask)
      x = xt + x
    return x if x_mask is None else x * x_mask


class DDSConv: # Dialted and Depth-Separable Convolution
  def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):
    self.channels, self.kernel_size, self.n_layers, self.p_dropout = channels, kernel_size, n_layers, p_dropout
    self.convs_sep, self.convs_1x1, self.norms_1, self.norms_2 = [], [], [], []
    for i in range(n_layers):
      dilation = kernel_size ** i
      padding = (kernel_size * dilation - dilation) // 2
      self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding))
      self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
      self.norms_1.append(LayerNorm(channels))
      self.norms_2.append(LayerNorm(channels))

  def forward(self, x, x_mask, g=None):
    if g is not None: x = x + g
    for i in range(self.n_layers):
      y = self.convs_sep[i](x * x_mask)
      y = self.norms_1[i].forward(y).gelu()
      y = self.convs_1x1[i](y)
      y = self.norms_2[i].forward(y).gelu()
      x = x + y.dropout(self.p_dropout)
    return x * x_mask


class ConvFlow:
  def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
    self.in_channels, self.filter_channels, self.kernel_size, self.n_layers, self.num_bins, self.tail_bound = in_channels, filter_channels, kernel_size, n_layers, num_bins, tail_bound
    self.half_channels = in_channels // 2
    self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
    self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.)
    self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = split_tinygrad(x, [self.half_channels]*2, 1)
    h = self.proj(self.convs.forward(self.pre(x0), x_mask, g=g)) * x_mask
    b, c, t = x0.shape
    h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2) # [b, cx?, t] -> [b, c, t, ?]
    unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
    unnormalized_heights = h[..., self.num_bins:2*self.num_bins] / math.sqrt(self.filter_channels)
    unnormalized_derivatives = h[..., 2 * self.num_bins:]
    x1, logabsdet = piecewise_rational_quadratic_transform(x1, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=reverse, tails='linear', tail_bound=self.tail_bound)
    x = x0.cat(x1, dim=1) * x_mask
    return x if reverse else (x, Tensor.sum(logabsdet * x_mask, [1,2]))

class ResidualCouplingLayer:
  def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
    assert channels % 2 == 0, "channels should be divisible by 2"
    self.channels, self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.mean_only = channels, hidden_channels, kernel_size, dilation_rate, n_layers, mean_only
    self.half_channels = channels // 2
    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = split_tinygrad(x, [self.half_channels]*2, 1)
    h = self.pre(x0) * x_mask
    h = self.enc.forward(h, x_mask, g=g)
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = split_tinygrad(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = Tensor.zeros_like(m)
    if not reverse:
      x1 = (m + x1 * logs.exp() * x_mask).realize()  # have to realize first or cat will merge the log op which causes nan (https://github.com/tinygrad/tinygrad/issues/1221)
      return x0.cat(x1, dim=1), logs.sum([1,2])
    else:
      x1 = ((x1 - m) * (-logs).exp() * x_mask).realize() # have to realize first or cat will merge the log op which causes nan (https://github.com/tinygrad/tinygrad/issues/1221)
      x = x0.cat(x1, dim=1)
      return x

class Log:
  def forward(self, x : Tensor, x_mask, reverse=False):
    if not reverse:
      y = x.maximum(1e-5).log() * x_mask
      return y, (-y).sum([1, 2])
    else: return x.exp() * x_mask

class Flip:
  def forward(self, x: Tensor, *args, reverse=False, **kwargs):
    x = x.flip([1])
    return x if reverse else (x, Tensor.zeros(x.shape[0], dtype=x.dtype).to(device=x.device))

class ElementwiseAffine:
  def __init__(self, channels):
    self.m, self.logs = Tensor.zeros(channels, 1), Tensor.zeros(channels, 1)
  def forward(self, x, x_mask, reverse=False, **kwargs): # x if reverse else y, logdet
      return (x - self.m) * Tensor.exp(-self.logs) * x_mask if reverse \
        else ((self.m + Tensor.exp(self.logs) * x) * x_mask, Tensor.sum(self.logs * x_mask, [1, 2]))