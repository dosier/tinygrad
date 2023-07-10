import numpy
import numpy as np

from tinygrad.ops import BinaryOps
from tinygrad.tensor import Tensor

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def piecewise_rational_quadratic_transform(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False, tails=None, tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=DEFAULT_MIN_DERIVATIVE):
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = cpu_unconstrained_rational_quadratic_spline
        spline_kwargs = {'tails': tails, 'tail_bound': tail_bound}
    outputs, logabsdet = spline_fn( inputs=inputs, unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights, unnormalized_derivatives=unnormalized_derivatives, inverse=inverse, min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative, **spline_kwargs)
    return outputs, logabsdet

def searchsorted(bin_locations, inputs, eps=1e-6):
  bin_locations[..., -1] += eps
  return np.sum(
    inputs[..., None] >= bin_locations,
    axis=-1
  ) - 1

def convert_pad_shape(pad_shape): return tuple(tuple(x) for x in pad_shape)

def tensor_unconstrained_rational_quadratic_spline(inputs: Tensor, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False, tails='linear', tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=DEFAULT_MIN_DERIVATIVE):
  inside_interval_mask = (inputs >= -tail_bound) * (inputs <= tail_bound)
  outside_interval_mask = (inside_interval_mask == False)

  outputs = numpy.zeros_like(inputs.shape)
  logabsdet = numpy.zeros_like(inputs.shape)
  unnormalized_derivatives = unnormalized_derivatives.numpy()
  if tails == 'linear':
    unnormalized_derivatives = numpy.pad(unnormalized_derivatives, ((0, 0), (0, 0), (0, 0), (1, 1)))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0
  else:
    raise RuntimeError('{} tails are not implemented.'.format(tails))
  outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
    inputs=inputs[inside_interval_mask], unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
    unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
    unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :], inverse=inverse, left=-tail_bound,
    right=tail_bound, bottom=-tail_bound, top=tail_bound, min_bin_width=min_bin_width, min_bin_height=min_bin_height,
    min_derivative=min_derivative)
  return outputs, logabsdet

def cpu_unconstrained_rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False, tails='linear', tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=DEFAULT_MIN_DERIVATIVE):
  # tensor_unconstrained_rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=inverse, tails=tails, tail_bound=tail_bound, min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative)
  inputs = inputs.numpy()
  inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
  outside_interval_mask = ~inside_interval_mask
  outputs = np.zeros_like(inputs)
  logabsdet = np.zeros_like(inputs)
  unnormalized_widths = unnormalized_widths.numpy()
  unnormalized_heights = unnormalized_heights.numpy()
  unnormalized_derivatives = unnormalized_derivatives.numpy()
  if tails == 'linear':
    unnormalized_derivatives = numpy.pad(unnormalized_derivatives, ((0, 0), (0, 0), (0, 0), (1, 1)))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0
  else:
    raise RuntimeError('{} tails are not implemented.'.format(tails))
  outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline( inputs=inputs[inside_interval_mask], unnormalized_widths=unnormalized_widths[inside_interval_mask, :], unnormalized_heights=unnormalized_heights[inside_interval_mask, :], unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :], inverse=inverse, left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound, min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative )
  return Tensor(outputs), Tensor(logabsdet)

def softmax(arr, dim=-1): return np.exp(arr) / np.sum(np.exp(arr), axis=dim, keepdims=True)  # softmax
def softplus(x): return np.log(1 + np.exp(x))

def rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False, left=0., right=1., bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH, min_bin_height=DEFAULT_MIN_BIN_HEIGHT, min_derivative=DEFAULT_MIN_DERIVATIVE):
    if numpy.min(inputs) < left or numpy.max(inputs) > right: raise ValueError('Input to a transform is not within its domain')
    num_bins = unnormalized_widths.shape[-1]
    if min_bin_width * num_bins > 1.0: raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0: raise ValueError('Minimal bin height too large for the number of bins')
    widths = softmax(unnormalized_widths, dim=-1) # softmax
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = widths.cumsum(axis=-1)
    cumwidths = numpy.pad(cumwidths, pad_width=((0, 0), (1,0)), mode='constant', constant_values=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + np.log1p(np.exp(unnormalized_derivatives))

    heights = softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = heights.cumsum(axis=-1)
    cumheights = numpy.pad(cumheights, pad_width=((0, 0), (1,0)), mode='constant', constant_values=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0], cumheights[..., -1] = bottom, top
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    bin_idx = searchsorted(cumheights if inverse else cumwidths, inputs)[..., None]

    input_cumwidths = np.take_along_axis(cumwidths, bin_idx, axis=-1)[..., 0]
    input_bin_widths = np.take_along_axis(widths, bin_idx, axis=-1)[..., 0]

    input_cumheights = np.take_along_axis(cumheights, bin_idx, axis=-1)[..., 0]
    delta = heights / widths
    input_delta = np.take_along_axis(delta, bin_idx, axis=-1)[..., 0]

    input_derivatives = np.take_along_axis(derivatives, bin_idx, axis=-1)[..., 0]
    input_derivatives_plus_one = np.take_along_axis(derivatives[..., 1:], bin_idx, axis=-1)[..., 0]

    input_heights = np.take_along_axis(heights, bin_idx, axis=-1)[..., 0]

    if inverse:
        a = ((inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta) + input_heights * (input_delta - input_derivatives))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)
        discriminant = numpy.square(b) - 4 * a * c
        assert (discriminant >= 0).all()
        root = (2 * c) / (-b - numpy.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = np.square(input_delta) * (input_derivatives_plus_one * np.square(root) + 2 * input_delta * theta_one_minus_theta + input_derivatives * np.square(1 - root))
        logabsdet = np.log(derivative_numerator) - 2 * np.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - theta).pow(2))
        logabsdet = derivative_numerator.log() - 2 * denominator.log()
        return outputs, logabsdet