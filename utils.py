import torch

from torch import nn, Tensor
from typing import Tuple


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """copied from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    :param dim1: for both src and tgt masking, this must be target sequence length
    :type dim1: int
    :param dim2: for src masking this must be encoder sequence length (i.e. 
    the length of the input sequence to the model), and for tgt masking, this must be target sequence length
    :type dim2: int
    :return: mask tensor of shape [dim1, dim2]
    :rtype: Tensor
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def get_src_trg(
        sequence: Tensor,
        enc_seq_len: int, 
        target_seq_len: int
        ) -> Tuple[Tensor, Tensor, Tensor]:
    """Slice the input sequences and return the appropriate tensors.
    Adapted from https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

    :param sequence: the input sequences of shape [seq_len, batch_size, embedding_dim]
    :type sequence: torch.Tensor
    :param enc_seq_len: the length of the historical data used for prediction
    :type enc_seq_len: int
    :param target_seq_len: the length of the target sequence
    :type target_seq_len: int
    :return: the tensors used for training src tgt and trg_y
    :rtype: Tuple[Tensor, Tensor, Tensor]
    """
    seq_len = sequence.shape[0]
    assert seq_len == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

    # encoder input
    src = sequence[:enc_seq_len, :, :]
    
    # decoder input. As per the paper, it must have the same dimension as the 
    # target sequence, and it must contain the last value of src, and all
    # values of trg_y except the last
    trg = sequence[enc_seq_len-1:seq_len-1, :, :]
    
    assert trg.shape[0] == target_seq_len, "Length of trg does not match target sequence length"

    # The target sequence against which the model output will be compared to compute loss
    trg_y = sequence[-target_seq_len:, :, :]

    assert trg_y.shape[0] == target_seq_len, "Length of trg_y does not match target sequence length"

    return src, trg, trg_y