import math

from examples.vits.debug import debug
from examples.vits.tiny.attention import Encoder
from examples.vits.tiny.custom_ops import split_tinygrad, squeeze_tinygrad
from examples.vits.tiny.modules import WN, ResBlock1, ResBlock2, LRELU_SLOPE, DDSConv, Log, ElementwiseAffine, ConvFlow, Flip, ResidualCouplingLayer, LayerNorm
from tinygrad.helpers import dtypes
from tinygrad.nn import Embedding, Conv1d, ConvTranspose1d
from tinygrad.tensor import Tensor


# PAPER: https://arxiv.org/abs/2106.06103
# CODE: https://github.com/jaywalnut310/vits/tree/main
# TODO: maybe use pre-existing attention mechanisms?


def sequence_mask(length: Tensor, max_length=None):
  if max_length is None:
    max_length = length.numpy().max()
  x = Tensor.arange(max_length, dtype=length.dtype, device=length.device)
  return Tensor(x.unsqueeze(0).numpy() < length.unsqueeze(1).numpy())
def slice_segments(x, ids_str, segment_size=4):
  ret = Tensor.zeros_like(x[:, :, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, :, idx_str:idx_end]
  return ret
def rand_slice_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = (Tensor.rand([b]).to(device=x.device) * ids_str_max).cast(dtype=dtypes.int64)
  ret = slice_segments(x, ids_str, segment_size)
  return ret, ids_str
def convert_pad_shape(pad_shape): return tuple(tuple(x) for x in pad_shape)
def generate_path(duration: Tensor, mask: Tensor):
  """
  duration: [b, 1, t_x]
  mask: [b, 1, t_y, t_x]
  """
  b, _, t_y, t_x = mask.shape
  cum_duration = duration.cumsum(axis = 2)
  cum_duration_flat = cum_duration.reshape(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).cast(mask.dtype)
  path = path.reshape(b, t_x, t_y)
  path = path - path.pad(convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  path = path.unsqueeze(1).transpose(2, 3) * mask
  return path

class StochasticDurationPredictor:
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels, self.filter_channels, self.kernel_size, self.p_dropout, self.n_flows, self.gin_channels = in_channels, filter_channels, kernel_size, p_dropout, n_flows, gin_channels
    self.log_flow, self.flows = Log(), [ElementwiseAffine(2)]
    for i in range(n_flows):
      self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(Flip())

    self.post_pre = Conv1d(1, filter_channels, 1)
    self.post_proj = Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = []
    self.post_flows.append(ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(Flip())
    self.pre = Conv1d(in_channels, filter_channels, 1)
    self.proj = Conv1d(filter_channels, filter_channels, 1)
    self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x: Tensor, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = self.pre(x.detach())
    if g is not None:
      x = x + self.cond(g.detach())
    x = self.convs.forward(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0
      h_w = self.post_pre(w)
      h_w = self.post_convs.forward(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = Tensor.randn(w.size(0), 2, w.size(2), dtype=x.dtype).to(device=x.device) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow.forward(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = z_q.split([1, 1], 1)
      u = z_u.sigmoid() * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += Tensor.sum((z_u.logsigmoid() + (-z_u).logsigmoid()) * x_mask, [1,2])
      logq = Tensor.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q
      logdet_tot = 0
      z0, logdet = self.log_flow.forward(z0, x_mask)
      logdet_tot += logdet
      z = z0.cat(z1, 1)
      for flow in flows:
        z, logdet = flow.forward(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = Tensor.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = Tensor.randn(x.shape[0], 2, x.shape[2], dtype=x.dtype).to(device=x.device) * noise_scale
      for flow in flows:
        z = flow.forward(z, x_mask, g=x, reverse=reverse)
      z0, z1 = split_tinygrad(z,[1, 1], 1)
      logw = z0
      return logw

class DurationPredictor:
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    self.in_channels, self.filter_channels, self.kernel_size, self.p_dropout, self.gin_channels = in_channels, filter_channels, kernel_size, p_dropout, gin_channels
    self.conv_1, self.norm_1 = Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2), LayerNorm(filter_channels)
    self.conv_2, self.norm_2 = Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2), LayerNorm(filter_channels)
    self.proj = Conv1d(filter_channels, 1, 1)
    if gin_channels != 0: self.cond = Conv1d(gin_channels, in_channels, 1)

  def forward(self, x: Tensor, x_mask, g=None):
    x = x.detach()
    if g is not None: x = x + self.cond(g.detach())
    x = self.conv_1(x * x_mask).relu()
    x = self.norm_1(x).dropout(self.p_dropout)
    x = self.conv_2(x * x_mask).relu(x)
    x = self.norm_2(x).dropout(self.p_dropout)
    return self.proj(x * x_mask) * x_mask

class TextEncoder:
  def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
    self.n_vocab, self.out_channels, self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout = n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
    self.emb = Embedding(n_vocab, hidden_channels)
    self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
    self.proj = Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x: Tensor, x_lengths: Tensor):
    debug(x, f"TextEncoder.forward[0](x)")
    x = self.emb(x)
    debug(x, f"TextEncoder.forward[1](x)")
    x = (x * math.sqrt(self.hidden_channels)).transpose(1, -1)  # [b, t, h] -transpose-> [b, h, t]
    debug(x, f"TextEncoder.forward[2](x)")
    x_mask = sequence_mask(x_lengths, x.shape[2]).unsqueeze(1).cast(x.dtype)  # TODO: verify this cast works
    debug(x_mask, f"TextEncoder.forward[3](x_mask)")
    x = self.encoder.forward(x * x_mask, x_mask)
    debug(x, f"TextEncoder.forward[4](x)")
    stats = self.proj(x) * x_mask
    debug(stats, f"TextEncoder.forward[5](stats)")
    m, logs = split_tinygrad(stats, self.out_channels, dim=1)
    debug(m, f"TextEncoder.forward[6](m)")
    return x, m, logs, x_mask


class ResidualCouplingBlock:
  def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
    self.channels, self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.n_flows, self.gin_channels = channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows, gin_channels
    self.flows = []
    for i in range(n_flows):
      self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows: x, _ = flow.forward(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows): x = flow.forward(x, x_mask, g=g, reverse=reverse)
    return x

class PosteriorEncoder:
  def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
    self.in_channels, self.out_channels, self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers, self.gin_channels = in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels
    self.pre = Conv1d(in_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).cast(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc.forward(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = stats.split(self.out_channels, dim=1)
    z = (m + Tensor.randn(m.shape, m.dtype) * logs.exp()) * x_mask
    return z, m, logs, x_mask

class Generator:
  def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
    self.num_kernels = len(resblock_kernel_sizes)
    self.num_upsamples = len(upsample_rates)
    self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
    resblock = ResBlock1 if resblock == '1' else ResBlock2
    self.ups = []
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
        self.ups.append(ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),k, u, padding=(k-u)//2))
    self.resblocks = []
    for i in range(len(self.ups)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
        self.resblocks.append(resblock(ch, k, d))
    self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
    if gin_channels != 0: self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

  def forward(self, x: Tensor, g=None):
    x = self.conv_pre(x)
    debug(x, f"Generator.forward(x)")
    if g is not None:  x = x + self.cond(g)
    for i in range(self.num_upsamples):
      x, xs = self.ups[i](x.leakyrelu(LRELU_SLOPE)), None
      debug(x, f"Generator.forward[{i}][0](x)")
      for j in range(self.num_kernels):
        if xs is None:
          xs = self.resblocks[i * self.num_kernels + j].forward(x)
        else:
          xs += self.resblocks[i * self.num_kernels + j].forward(x)
      x = xs / self.num_kernels
      debug(x, f"Generator.forward[{i}][1](x)")
      debug(xs, f"Generator.forward[{i}][1](xs)")
    x = x.leakyrelu()
    debug(x, f"Generator.forward1(x)")
    x = self.conv_post(x)
    debug(x, f"Generator.forward2(x)")
    x = x.tanh()
    debug(x, f"Generator.forward3(x)")
    return x

class SynthesizerTrn:  # Synthesizer for Training
  def __init__(self, n_vocab, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, n_speakers=0, gin_channels=0, use_sdp=True, **kwargs):
    self.n_vocab, self.spec_channels, self.inter_channels, self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.resblock, self.resblock_kernel_sizes, self.resblock_dilation_sizes, self.upsample_rates, self.upsample_initial_channel, self.upsample_kernel_sizes, self.segment_size, self.n_speakers, self.gin_channels, self.use_sdp = n_vocab, spec_channels, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, segment_size, n_speakers, gin_channels, use_sdp
    self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
    self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels) if use_sdp else DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)
    if n_speakers > 1: self.emb_g = Embedding(n_speakers, gin_channels)

  def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    debug(x, f"SynthesizerTrn.infer[0](x)")
    debug(x_lengths, f"SynthesizerTrn.infer[0](x_lengths)")
    x, m_p, logs_p, x_mask = self.enc_p.forward(x, x_lengths)
    debug(x, f"SynthesizerTrn.infer[1](x)")
    debug(m_p, f"SynthesizerTrn.infer[1](m_p)")
    debug(logs_p, f"SynthesizerTrn.infer[1](logs_p)")
    debug(x_mask, f"SynthesizerTrn.infer[1](x_mask)")
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None
    if self.use_sdp:
      logw = self.dp.forward(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.dp.forward(x, x_mask, g=g)
    debug(logw, f"SynthesizerTrn.infer[2](logw)")
    w = Tensor.exp(logw) * x_mask * length_scale
    debug(w, f"SynthesizerTrn.infer[2](w)")
    w_ceil = Tensor.ceil(w)
    debug(w_ceil, f"SynthesizerTrn.infer[2](w_ceil)")
    y_lengths = Tensor.maximum(w_ceil.sum([1, 2]), 1).cast(dtypes.int64)
    debug(y_lengths, f"SynthesizerTrn.infer[2](y_lengths)")
    y_mask = sequence_mask(y_lengths, None).unsqueeze(1).cast(x_mask.dtype)
    debug(y_mask, f"SynthesizerTrn.infer[2](y_mask)")
    attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
    debug(attn_mask, f"SynthesizerTrn.infer[2](attn_mask)")
    attn = generate_path(w_ceil, attn_mask)
    debug(attn, f"SynthesizerTrn.infer[2](attn)")
    m_p = squeeze_tinygrad(attn, 1).matmul(m_p.transpose(1, 2)).transpose(1, 2)       # [b, t', t], [b, t, d] -> [b, d, t']
    debug(m_p, f"SynthesizerTrn.infer[3](m_p)")
    logs_p = squeeze_tinygrad(attn, 1).matmul(logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    debug(logs_p, f"SynthesizerTrn.infer[3](logs_p)")
    z_p = m_p + Tensor.randn(*m_p.shape, dtype=m_p.dtype) * logs_p.exp() * noise_scale
    debug(z_p, f"SynthesizerTrn.infer[3](z_p)")
    y_mask = y_mask.cast(z_p.dtype)
    z = self.flow.forward(z_p, y_mask, g=g, reverse=True)
    debug(z, f"SynthesizerTrn.infer[3](z)")
    o = self.dec.forward((z * y_mask)[:, :, :max_len], g=g)
    debug(o, f"SynthesizerTrn.infer[3](o)")
    return o, attn, y_mask, (z, z_p, m_p, logs_p)