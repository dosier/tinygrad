from pathlib import Path

import numpy as np

from examples.vits.tiny import utils
from examples.vits.tiny.models import SynthesizerTrn
from examples.vits.tiny.text.symbols import symbols
from examples.vits.tiny.text import text_to_sequence
from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor
import IPython.display as ipd

def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result

def get_text(text, hps):
  text_norm = text_to_sequence(text, hps.data.text_cleaners)
  if hps.data.add_blank: text_norm = intersperse(text_norm, 0)
  text_norm = Tensor(text_norm, dtype=dtypes.int64)
  return text_norm

# TODO: auto download from drive https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2
def load_model():
  hps = utils.get_hparams_from_file(Path(__file__).parent / "../configs/ljs_base.json")
  net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
  )
  _ = utils.load_checkpoint(Path(__file__).parent / "../weights/pretrained_ljs.pth", net_g, None)
  return net_g, hps

if __name__ == '__main__':
  Tensor.no_grad = True
  Tensor.Training = False
  Tensor.manual_seed(1337)  # Deterministic
  np.random.seed(1337)
  net_g, hps = load_model()
  stn_tst = get_text("VITS is Awesome!", hps)
  print(f"Text({stn_tst.shape}): {stn_tst.numpy()}")
  x_tst = stn_tst.unsqueeze(0)
  x_tst_lengths = Tensor([stn_tst.shape[0]], dtype=dtypes.int64)
  out = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)
  audio = out[0][0, 0].numpy()
  audio = ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False)
  with open('test.wav', 'wb') as f:
    f.write(audio.data)