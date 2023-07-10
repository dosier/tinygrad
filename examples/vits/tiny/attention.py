import math

import numpy.random

from examples.vits.debug import debug
from examples.vits.tiny.custom_ops import masked_fill_tinygrad
from examples.vits.tiny.modules import LayerNorm
from tinygrad.nn import Conv1d
from tinygrad.tensor import Tensor


def convert_pad_shape(pad_shape): return tuple(tuple(x) for x in pad_shape)

class MultiHeadAttention:
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    assert channels % n_heads == 0
    self.channels, self.out_channels, self.n_heads, self.p_dropout, self.window_size, self.heads_share, self.block_length, self.proximal_bias, self.proximal_init = channels, out_channels, n_heads, p_dropout, window_size, heads_share, block_length, proximal_bias, proximal_init
    self.attn = None
    self.k_channels = channels // n_heads
    self.conv_q, self.conv_k, self.conv_v = Conv1d(channels, channels, 1), Conv1d(channels, channels, 1), Conv1d(channels, channels, 1)
    self.conv_o = Conv1d(channels, out_channels, 1)
    if window_size is not None:
      self.emb_rel_k, self.emb_rel_v = [Tensor.randn(1 if heads_share else n_heads, window_size * 2 + 1, self.k_channels) * (self.k_channels ** -0.5)] * 2

  def forward(self, x, c, attn_mask=None):
    q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
    debug(q, "MultiHeadAttention.forward[0](q)")
    debug(k, "MultiHeadAttention.forward[0](k)")
    debug(v, "MultiHeadAttention.forward[0](v)")
    x, self.attn = self.attention(q, k, v, mask=attn_mask)
    debug(x, "MultiHeadAttention.forward[0](x)")
    x = self.conv_o(x)
    debug(x, "MultiHeadAttention.forward[1](x)")
    return x

  def attention(self, query: Tensor, key: Tensor, value: Tensor, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = key.shape[0], key.shape[1], key.shape[2], query.shape[2]
    query = query.reshape(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.reshape(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.reshape(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    scores : Tensor = (query / math.sqrt(self.k_channels)) @ key.transpose(-2, -1)
    debug(scores, "MultiHeadAttention.attention[0](scores)")
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      debug(key_relative_embeddings, "MultiHeadAttention.attention[1](key_relative_embeddings)")
      rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
      debug(rel_logits, "MultiHeadAttention.attention[1](rel_logits)")
      scores_local = self._relative_position_to_absolute_position(rel_logits)
      debug(scores_local, "MultiHeadAttention.attention[1](scores_local)")
      scores = scores + scores_local
      debug(scores, f"MultiHeadAttention.attention[1](scores)")
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = masked_fill_tinygrad(scores, mask == 0, -1e4)
      if self.block_length is not None:
        assert t_s == t_t, "Local attention is only available for self-attention."
        block_mask = Tensor.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = masked_fill_tinygrad(scores, block_mask == 0, -1e4)
    debug(scores, "MultiHeadAttention.attention[2](scores)")
    p_attn = scores.softmax(axis=-1)  # [b, n_h, t_t, t_s]
    debug(p_attn, "MultiHeadAttention.attention[2](p_attn)")
    output = p_attn.matmul(value)
    debug(output, "MultiHeadAttention.attention[3](output)")
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().reshape(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn
  def _matmul_with_relative_values(self, x, y): return x.matmul(y.unsqueeze(0))                 # x: [b, h, l, m], y: [h or 1, m, d], ret: [b, h, l, d]
  def _matmul_with_relative_keys(self, x, y): return x.matmul(y.unsqueeze(0).transpose(-2, -1)) # x: [b, h, l, d], y: [h or 1, m, d], re, : [b, h, l, m]
  def _get_relative_embeddings(self, relative_embeddings, length):
    pad_length, slice_start_position = max(length - (self.window_size + 1), 0), max((self.window_size + 1) - length, 0)
    padded_relative_embeddings = relative_embeddings if pad_length <= 0\
      else relative_embeddings.pad(convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    return padded_relative_embeddings[:, slice_start_position:(slice_start_position + 2 * length - 1)] #used_relative_embeddings
  def _relative_position_to_absolute_position(self, x: Tensor):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.shape
    # Concat columns of pad to shift from relative to absolute indexing.
    x = x.pad(convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))
    debug(x, "MultiHeadAttention._relative_position_to_absolute_position[0](x)")
    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.reshape([batch, heads, length * 2 * length])
    debug(x_flat, "MultiHeadAttention._relative_position_to_absolute_position[1](x_flat)")
    x_flat = x_flat.pad(convert_pad_shape([[0,0],[0,0],[0,length-1]]))
    debug(x_flat, "MultiHeadAttention._relative_position_to_absolute_position[2](x_flat)")
    # Reshape and slice out the padded elements.
    x_final = x_flat.reshape([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    debug(x_final, "MultiHeadAttention._relative_position_to_absolute_position[3](x_final)")
    return x_final
  def _absolute_position_to_relative_position(self, x: Tensor):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.shape
    # padd along column
    x = x.pad(convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.reshape([batch, heads, length**2 + length*(length -1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = x_flat.pad(convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.reshape([batch, heads, length, 2*length])[:,:,:,1:]
    return x_final


class FFN:
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, causal=False):
    self.in_channels, self.out_channels, self.filter_channels, self.kernel_size, self.p_dropout, self.activation, self.causal = in_channels, out_channels, filter_channels, kernel_size, p_dropout, activation, causal
    self.padding = self._causal_padding if causal else self._same_padding
    self.conv_1, self.conv_2 = Conv1d(in_channels, filter_channels, kernel_size), Conv1d(filter_channels, out_channels, kernel_size)
  def forward(self, x, x_mask):
    x = self.conv_1(self.padding(x * x_mask))
    x = x * (1.702 * x).sigmoid() if self.activation == "gelu" else x.relu()
    x = x.dropout(self.p_dropout)
    x = self.conv_2(self.padding(x * x_mask))
    return x * x_mask
  def _causal_padding(self, x):
    if self.kernel_size == 1: return x
    return x.pad(convert_pad_shape([[0, 0], [0, 0], [self.kernel_size - 1, 0]]))
  def _same_padding(self, x):
    if self.kernel_size == 1: return x
    return x.pad(convert_pad_shape([[0, 0], [0, 0], [(self.kernel_size - 1) // 2, self.kernel_size // 2]]))

class Encoder:
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
    self.hidden_channels, self.filter_channels, self.n_heads, self.n_layers, self.kernel_size, self.p_dropout, self.window_size = hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, window_size
    self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2 = [], [], [], []
    for _ in range(n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    debug(attn_mask, "Encoder.forward[0](attn_mask)")
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.attn_layers[i].forward(x, x, attn_mask)
      debug(y, f"Encoder.forward[{i}][0](y)")
      y = y.dropout(self.p_dropout)
      x = self.norm_layers_1[i].forward(x + y)
      debug(x, f"Encoder.forward[{i}][1](x)")
      y = self.ffn_layers[i].forward(x, x_mask)
      debug(y, f"Encoder.forward[{i}][2](y)")
      y = y.dropout(self.p_dropout)
      x = self.norm_layers_2[i].forward(x + y)
      debug(x, f"Encoder.forward[{i}][3](x)")

    x = x * x_mask
    return x