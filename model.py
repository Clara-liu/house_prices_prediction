import torch
import math

import numpy as np
from torch import nn, Tensor
from typing import Tuple


class PositionalEncoding(nn.Module):
    # copied from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # note the input dimension specification for the forward method
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]  # encoding on the sequence length dim
        return self.dropout(x)
    


def get_src_trg(
        sequence: torch.Tensor, 
        enc_seq_len: int, 
        target_seq_len: int
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Slice the input sequences and return the appropriate tensors.
    Adapted from https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

    :param sequence: the input sequences of shape [seq_len, batch_size, embedding_dim]
    :type sequence: torch.Tensor
    :param enc_seq_len: the length of the historical data used for prediction
    :type enc_seq_len: int
    :param target_seq_len: the length of the target sequence
    :type target_seq_len: int
    :return: the tensors used for training src tgt and trg_y
    :rtype: Tuple[torch.tensor, torch.tensor, torch.tensor]
    """
    seq_len = sequence.shape[0]
    assert seq_len == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

    # encoder input
    src = sequence[:enc_seq_len, :, :]
    
    # decoder input. As per the paper, it must have the same dimension as the 
    # target sequence, and it must contain the last value of src, and all
    # values of trg_y except the last (i.e. it must be shifted right by 1)
    trg = sequence[enc_seq_len-1:seq_len-1, :, :]
    
    assert trg.shape[0] == target_seq_len, "Length of trg does not match target sequence length"

    # The target sequence against which the model output will be compared to compute loss
    trg_y = sequence[-target_seq_len:, :, :]

    assert trg_y.shape[0] == target_seq_len, "Length of trg_y does not match target sequence length"

    return src, trg, trg_y # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]