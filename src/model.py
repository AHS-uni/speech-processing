"""
ASR model architecture definitions.

This module provides:
- `HybridASR`: combines a convolutional subsampler, Transformer encoder, CTC head, and Transformer decoder for ASR.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class HybridASR(nn.Module):
    """
    Hybrid ASR model with CTC and autoregressive decoding.

    Args:
        cfg (dict): config dict containing:
            - 'n_mels': number of Mel bins
            - 'vocab_size': size of tokenizer vocabulary
        d_model (int): feature dimension for encoder/decoder
        nhead (int): number of attention heads
        num_encoder_layers (int): number of encoder layers
        num_decoder_layers (int): number of decoder layers
        dim_feedforward (int): inner dimension of feedforward layers
        dropout (float): dropout rate
    """

    def __init__(
        self,
        cfg: dict,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = cfg["vocab_size"]
        n_mels = cfg["n_mels"]
        # convolutional subsampler: collapse freq dim into channels
        self.subsampler = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=(n_mels, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.ReLU(),
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        # CTC head
        self.ctc_head = nn.Linear(d_model, self.vocab_size)
        # embedding for decoder input tokens
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        # output projection for decoder
        self.decoder_fc = nn.Linear(d_model, self.vocab_size)

    def forward(
        self, specs: torch.Tensor, tokens: Optional[torch.LongTensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            specs (Tensor): [batch, 1, n_mels, T]
            tokens (Optional[LongTensor]): [batch, L] input tokens for teacher forcing.

        Returns:
            Union[
                Tensor,                        # CTC-only: [T', batch, vocab_size]
                Tuple[Tensor, Tensor]          # (CTC, Dec): ([T', batch, vocab_size], [batch, L, vocab_size])
            ]
        """
        # subsample and collapse freq dim
        x = self.subsampler(specs)  # [B, d_model, 1, T']
        x = x.squeeze(2)  # [B, d_model, T']
        # prepare for Transformer: [T', B, d_model]
        x = x.permute(2, 0, 1)
        # encoder
        memory = self.encoder(x)  # [T', B, d_model]
        # CTC logits
        ctc_logits = self.ctc_head(memory)  # [T', B, vocab_size]
        if tokens is None:
            return ctc_logits
        # decoder path with teacher forcing
        tgt = self.embedding(tokens)  # [B, L, d_model]
        tgt = tgt.permute(1, 0, 2)  # [L, B, d_model]
        # generate causal mask for decoder
        L = tgt.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(specs.device)
        # decoder
        dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # [L, B, d_model]
        dec_out = dec_out.permute(1, 0, 2)  # [B, L, d_model]
        # project to vocab
        dec_logits = self.decoder_fc(dec_out)  # [B, L, vocab_size]
        return ctc_logits, dec_logits
