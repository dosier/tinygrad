import json
import logging
import os
import sys

import numpy as np

from examples.vits.tiny.custom_ops import weight_norm_tinygrad
from examples.vits.tiny.models import SynthesizerTrn
from examples.vits.tiny.modules import LayerNorm
from tinygrad import nn
from tinygrad.state import torch_load
from tinygrad.tensor import Tensor

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def load_checkpoint(checkpoint_path, model: SynthesizerTrn, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch_load(checkpoint_path)
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  weight_g, weight_v = None, None
  parent = None
  for key, v in saved_state_dict.items():
    # print(key, v.shape)

    try:
      obj = model
      skip = False
      for k in key.split('.'):
        if k.isnumeric(): obj = obj[int(k)]
        elif isinstance(obj, dict): obj = obj[k]
        else:
          if isinstance(obj, LayerNorm) or isinstance(obj, nn.LayerNorm):
            if k == "gamma": k = "weight"
            if k == "beta": k = "bias"
          elif k == "weight_g":
            parent = obj
            weight_g = v
            skip = True
          elif k == "weight_v":
            weight_v = v
            skip = True
          if not skip: obj = getattr(obj, k)

      if weight_g is not None or weight_v is not None:
        if weight_g is not None and weight_v is not None:
          # print(f"setting weight of g, v from {parent}")
          setattr(obj, "weight_g", weight_g.numpy())
          setattr(obj, "weight_v", weight_v.numpy())
          obj = getattr(parent, "weight")
          v = weight_norm_tinygrad(weight_v, weight_g, 0)
          weight_g, weight_v = None, None
          parent = None
          skip = False
      if not skip:
        if obj.shape == v.shape:
          obj.assign(v.to(obj.device))
        else:
          logger.error("MISMATCH SHAPE IN %s, %r %r" % (key, obj.shape, v.shape))
    except Exception as e:
      logger.error("EXCEPTION IN %s" % key)
      logger.error(e)
      raise e
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration

def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  hparams =HParams(**json.loads(data))
  return hparams

class HParams:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict: v = HParams(**v)
      self[k] = v
  def keys(self): return self.__dict__.keys()
  def items(self): return self.__dict__.items()
  def values(self): return self.__dict__.values()
  def __len__(self): return len(self.__dict__)
  def __getitem__(self, key): return getattr(self, key)
  def __setitem__(self, key, value): return setattr(self, key, value)
  def __contains__(self, key): return key in self.__dict__
  def __repr__(self): return self.__dict__.__repr__()
