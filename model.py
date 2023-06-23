import torch
import math

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


class TimeSeriesTransformer(nn.Module):
    """This time series transformer model is based on the paper by
    Wu et al (2020) and the blog post by Kasper G. A. Ludvigsen
    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

    :param nn: _description_
    :type nn: _type_
    """
    def __init__(
            self,
            enc_len: int,
            model_dim: int = 512,
            n_heads: int = 4,
            dropout_encoder: float=0.15,
            dropout_decoder: float=0.15,
            n_encoder_layers: int = 4,
            n_decoder_layers: int = 4,
            dim_feedforward_encoder: int=2048,
            dim_feedforward_decoder: int=2048
        ) -> None:
        super().__init__()
        # embed univariate input
        self.encoder_input_layer = nn.Linear(in_features=1, out_features=model_dim)
        # position encoding
        self.position_encoder = PositionalEncoding(model_dim, max_len=enc_len)
        # encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            batch_first=False,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder
        )
        # encoder block
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )
        # decoder input layer from trg sequence
        self.decoder_input_layer = nn.Linear(in_features=1, out_features=model_dim)
        # decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            batch_first=False,
            dropout=dropout_decoder,
            dim_feedforward=dim_feedforward_decoder
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

        # final univariate output
        self.final_fc = nn.Linear(in_features=model_dim, out_features=1)

    def forward(
            self,
            src: Tensor,
            trg: Tensor,
            src_mask: Tensor,
            trg_mask: Tensor
    ) -> Tensor:
        # encode
        src = self.encoder_input_layer(src)  # [seq_len, batch_size, input_feature_dim] -> [seq_len, batch_size, model_dim]
        print(f'After encoder embedding: src shape {src.shape}')
        src = self.position_encoder(src)  # [seq_len, batch_size, model_dim] -> [seq_len, batch_size, model_dim]
        print(f'After position encoding: src shape {src.shape}')
        src = self.encoder(src)  # no masking because no padding used for this data
        print(f'After encoder: src shape {src.shape}')
        # decode
        decoder_output = self.decoder_input_layer(trg)  # [trg_seq_len, batch_size. input_feature_dim] -> [trg_seq_len, batch_size, model_dim]
        print(f'After decoder embedding: decoder output shape {decoder_output.shape}')
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=trg_mask,
            memory_mask=src_mask
        )
        print(f'After decoder: decoder output shape {decoder_output.shape}')
        decoder_output = self.final_fc(decoder_output)
        print(f'After final linear layer: decoder output shape {decoder_output.shape}')

        return decoder_output



# ############ TRAIN TEST #############
# data = torch.rand(340, 32, 1)  # [total_seq_len, batch_size, input_feature_dim]

# model_dim = 512
# n_heads = 8
# enc_len = 330
# trg_len = 10

# model = TimeSeriesTransformer(enc_len=enc_len)

# src_mask = generate_square_subsequent_mask(
#     dim1=trg_len,
#     dim2=enc_len
# )
# trg_mask = generate_square_subsequent_mask(
#     dim1=trg_len,
#     dim2=trg_len
# )

# src, trg, trg_y = get_src_trg(data, enc_len, trg_len)

# output = model(src, trg, src_mask, trg_mask)

# print(output.shape)

# ####### INFERENCE TEST ##########

# src = torch.rand(330, 32, 1)
# trg = torch.rand(10, 32, 1)  # [seq_len, batch_size, feature_dim] note that the first input should be the last known value

# dim1 = trg.shape[0]
# dim2 = src.shape[0]

# trg_mask_inf = generate_square_subsequent_mask(
#     dim1=dim1,
#     dim2=dim1
#     )

# src_mask_inf = generate_square_subsequent_mask(
#     dim1=dim1,
#     dim2=dim2
#     )
# model.eval()
# pred = model(src, trg, src_mask_inf, trg_mask_inf)
# print(pred.shape)