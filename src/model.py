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
        input_size (int): Size of input feature dimension (e.g., number of Mel bins).
        output_size (int): Size of output vocabulary (e.g., tokenizer vocab size).
        d_model (int): Feature dimension for encoder/decoder.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of Transformer encoder layers.
        num_decoder_layers (int): Number of Transformer decoder layers.
        dim_feedforward (int): Feedforward network inner dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        input_size: int,  # number of Mel bins
        output_size: int,  # vocab size
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = output_size

        # Convolutional subsampler: collapse frequency dimension into channels
        self.subsampler = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=(input_size, 3),
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
        self.ctc_head = nn.Linear(d_model, output_size)

        # Embedding for decoder input tokens
        self.embedding = nn.Embedding(output_size, d_model)

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

        # Output projection for decoder
        self.decoder_fc = nn.Linear(d_model, output_size)

    def forward(
        self, specs: torch.Tensor, tokens: Optional[torch.LongTensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            specs (Tensor): [batch, 1, input_size, T]
            tokens (Optional[LongTensor]): [batch, L] input tokens for teacher forcing.

        Returns:
            - CTC logits: [T', batch, vocab_size]
            - (Optional) Decoder logits: [batch, L, vocab_size]
        """
        x = self.subsampler(specs)  # [B, d_model, 1, T']
        x = x.squeeze(2)  # [B, d_model, T']
        x = x.permute(2, 0, 1)  # [T', B, d_model]

        memory = self.encoder(x)  # [T', B, d_model]
        ctc_logits = self.ctc_head(memory)  # [T', B, vocab_size]

        if tokens is None:
            return ctc_logits

        # Decoder with teacher forcing
        tgt = self.embedding(tokens).permute(1, 0, 2)  # [L, B, d_model]
        L = tgt.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(specs.device)

        dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # [L, B, d_model]
        dec_logits = self.decoder_fc(dec_out.permute(1, 0, 2))  # [B, L, vocab_size]

        return ctc_logits, dec_logits
