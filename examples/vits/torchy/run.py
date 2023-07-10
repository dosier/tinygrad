from pathlib import Path

import numpy as np
import torch

from examples.vits.tiny.text import text_to_sequence
from examples.vits.tiny.text.symbols import symbols
from examples.vits.torchy import utils, commons
from examples.vits.torchy.models import SynthesizerTrn
import IPython.display as ipd

def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
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
  # torch.set_default_device("mps")
  torch.manual_seed(1337)  # Deterministic
  np.random.seed(1337)
  net_g, hps = load_model()
  stn_tst = get_text("VITS is Awesome!", hps)
  # print(stn_tst.dtype)
  with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
      0, 0].data.cpu().float().numpy()
  ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))