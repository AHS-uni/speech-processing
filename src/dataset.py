"""
LJSpeech dataset loader and batching utilities.

This module provides:
- `LJSpeechDataset`: a PyTorch `Dataset` loading precomputed spectrograms and token sequences per split.
- `collate_fn`: a function to batch and pad variable-length spectrograms and token sequences.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Collate function for LJSpeechDataset items (dicts).
    Batch is a list of dicts with keys:
      - 'specs': Tensor [1, n_mels, T]
      - 'spec_lengths': scalar Tensor
      - 'tokens': LongTensor [L]
      - 'token_lengths': scalar Tensor
    Returns a dict of batched tensors:
      - 'specs': FloatTensor [B, 1, n_mels, T_max]
      - 'spec_lengths': LongTensor [B]
      - 'tokens': LongTensor [B, L_max]
      - 'token_lengths': LongTensor [B]
    """
    specs = [
        item["specs"].squeeze(0).transpose(0, 1) for item in batch
    ]  # list of [T, n_mels]
    spec_lengths = torch.tensor([s.shape[0] for s in specs], dtype=torch.long)
    # pad along time dimension (dim=0)
    padded_specs = pad_sequence(specs, batch_first=True)  # [B, T_max, n_mels]
    # restore shape [B,1,n_mels,T_max]
    padded_specs = padded_specs.transpose(1, 2).unsqueeze(1)

    tokens = [item["tokens"] for item in batch]
    token_lengths = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)
    padded_tokens = pad_sequence(
        tokens, batch_first=True, padding_value=0
    )  # [B, L_max]

    return {
        "specs": padded_specs,
        "spec_lengths": spec_lengths,
        "tokens": padded_tokens,
        "token_lengths": token_lengths,
    }


class LJSpeechDataset(Dataset):
    """
    PyTorch Dataset for preprocessed LJSpeech ASR data.

    Each item is a dict with:
        - 'specs': FloatTensor [1, n_mels, T]
        - 'spec_lengths': LongTensor [1]
        - 'tokens': LongTensor [L]
        - 'token_lengths': LongTensor [1]

    Args:
        split (str): One of {'train','val','test'} indicating which split to load.
        preprocessed_dir (str): Directory containing {split}_data.pt bundles.
        transform (callable, optional): A function to apply to each spectrogram.
    """

    def __init__(
        self,
        split: str,
        preprocessed_dir: str,
        transform: callable = None,
    ):
        self.split = split
        self.transform = transform

        bundle_path = os.path.join(preprocessed_dir, f"{split}_data.pt")
        self.bundle = torch.load(bundle_path)

    def __len__(self) -> int:
        return len(self.bundle)

    def __getitem__(self, idx: int) -> dict:
        entry = self.bundle[idx]
        spec = entry["spec"]  # [n_mels, T]
        tokens = entry["tokens"]  # LongTensor [L]

        # Optional augmentation / transform
        if self.transform is not None:
            spec = self.transform(spec)

        # Specs need shape [1, n_mels, T]
        specs = spec.unsqueeze(0)

        # Lengths
        spec_len = specs.shape[-1]
        token_len = tokens.shape[0]

        return {
            "specs": specs,  # [1, n_mels, T]
            "spec_lengths": torch.tensor(spec_len),  # scalar
            "tokens": tokens.long(),  # [L]
            "token_lengths": torch.tensor(token_len),  # scalar
        }
