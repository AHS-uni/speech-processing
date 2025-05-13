import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class PositionalEncoding(nn.Module):
    """
    Positional encoding module injects information about the relative or absolute
    position of the tokens in the sequence. The positional encodings have the
    same dimension as the embeddings so that the two can be summed.

    Args:
        d_model (int): Embedding size (feature dimension).
        max_len (int): Maximum sequence length supported.

    Shapes:
        x: [T, B, d_model]
        output: [T, B, d_model] (same as input)
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to the input tensor.

        Args:
            x (Tensor): Input tensor of shape [T, B, d_model].

        Returns:
            Tensor of same shape with positional encodings added.
        """
        T = x.size(0)
        x = x + self.pe[:T]
        return x


class HybridASR(nn.Module):
    """
    Improved Hybrid ASR model combining a convolutional subsampler,
    Transformer encoder with positional encoding, CTC head, and
    Transformer decoder for autoregressive decoding.

    Args:
        input_size (int): Number of Mel-frequency bins (input feature dim).
        output_size (int): Vocabulary size for tokens.
        d_model (int): Model dimension for all internal representations.
        nhead (int): Number of attention heads in multihead attention.
        num_encoder_layers (int): Number of layers in the Transformer encoder.
        num_decoder_layers (int): Number of layers in the Transformer decoder.
        dim_feedforward (int): Inner dimension of the feedforward network.
        dropout (float): Dropout probability throughout the model.

    Shapes:
        specs: [B, 1, input_size, T]           (batch, channel, freq, time)
        ctc_logits: [T', B, output_size]       (time', batch, vocab)
        dec_logits: [B, L, output_size]        (batch, seq_len, vocab)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = output_size

        # Convolutional subsampler: two conv blocks
        self.subsampler = nn.Sequential(
            nn.Conv2d(
                1, d_model, kernel_size=(input_size, 3), stride=(1, 2), padding=(0, 1)
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                d_model, d_model, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # total subsample rate is 2 * 2 = 4
        self.subsample_rate = (
            self.subsampler[0].stride[1] * self.subsampler[3].stride[1]
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder with increased capacity
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

        # CTC head for alignment-based loss
        self.ctc_head = nn.Linear(d_model, output_size)

        # Embedding + Transformer decoder for seq2seq
        self.embedding = nn.Embedding(output_size, d_model)
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
        self.decoder_fc = nn.Linear(d_model, output_size)

    def forward(
        self, specs: torch.Tensor, tokens: Optional[torch.LongTensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the ASR model.

        Args:
            specs (Tensor): Mel-spectrogram input [B, 1, input_size, T].
            tokens (LongTensor, optional): Token IDs for teacher forcing [B, L].

        Returns:
            ctc_logits (Tensor): [T', B, vocab_size] from CTC head.
            dec_logits (Tensor): [B, L, vocab_size] from decoder (if tokens provided).
        """
        # Subsampling and feature extraction
        x = self.subsampler(specs)  # [B, d_model, 1, T']
        x = x.squeeze(2)  # [B, d_model, T']
        x = x.permute(2, 0, 1)  # [T', B, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder + CTC head
        memory = self.encoder(x)  # [T', B, d_model]
        ctc_logits = self.ctc_head(memory)  # [T', B, vocab_size]

        if tokens is None:
            return ctc_logits

        # Autoregressive decoding with teacher forcing
        tgt = self.embedding(tokens).permute(1, 0, 2)  # [L, B, d_model]
        L = tgt.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(specs.device)
        dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # [L, B, d_model]
        dec_logits = self.decoder_fc(dec_out.permute(1, 0, 2))  # [B, L, vocab_size]

        return ctc_logits, dec_logits
