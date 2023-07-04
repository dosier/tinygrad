import numpy
import numpy as np
import torch
import torchaudio as torchaudio

from examples.rnn_t.train_rnn_t import rnnt_loss, rnnt_loss_batch
from tinygrad.tensor import Tensor

def get_basic_data(device):
    # Example provided
    # in 6f73a2513dc784c59eec153a45f40bc528355b18
    # of https://github.com/HawkAaron/warp-transducer

    logits = torch.tensor(
        [
            [
                [
                    [0.1, 0.6, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.6, 0.1, 0.1],
                    [0.1, 0.1, 0.2, 0.8, 0.1],
                ],
                [
                    [0.1, 0.6, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.2, 0.1, 0.1],
                    [0.7, 0.1, 0.2, 0.1, 0.1],
                ],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor([[1, 2]], dtype=torch.int, device=device)
    logit_lengths = torch.tensor([2], dtype=torch.int, device=device)
    target_lengths = torch.tensor([2], dtype=torch.int, device=device)

    logits.requires_grad_(True)

    return logits, targets, logit_lengths, target_lengths


def to_tiny(x: torch.Tensor) -> Tensor:
  return x.detach().numpy()

if __name__ == "__main__":
  logits, targets, logit_lengths, target_lengths = get_basic_data('cpu')
  print(logits)
  print(logit_lengths)
  print(targets)
  print(target_lengths)
  logits = logits.log_softmax(dim=-1)
  torch_loss = torchaudio.functional.rnnt_loss(logits, targets, logit_lengths, target_lengths,
                                               reduction="none", fused_log_softmax=False, blank=-1)

  tiny_loss = rnnt_loss_batch(to_tiny(logits), to_tiny(logit_lengths), to_tiny(targets), to_tiny(target_lengths),
                              blank=-1)
  print(torch_loss.detach().numpy()[0])
  print(tiny_loss[0][0])
