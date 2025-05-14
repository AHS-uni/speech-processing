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
    Batch and pad spectrograms and token sequences.

    Args:
        batch (list of tuples): [(spec, tokens), …] where
            spec   is Tensor [n_mels, T_i]
            tokens is LongTensor [L_i]

    Returns:
        dict:
            specs         Tensor [B, 1, n_mels, T_max]
            spec_lengths  LongTensor [B]
            tokens        LongTensor [B, L_max]
            token_lengths LongTensor [B]
    """
    specs, tokens = zip(*batch)

    # ——— Token padding ———
    token_lengths = torch.tensor([t.numel() for t in tokens], dtype=torch.long)
    tokens_padded = pad_sequence(list(tokens), batch_first=True, padding_value=0)

    # ——— Spectrogram padding ———
    # Transpose to make time the first dim: [n_mels, T] → [T, n_mels]
    specs_t = [s.transpose(0, 1) for s in specs]
    spec_lengths = torch.tensor([t.shape[0] for t in specs_t], dtype=torch.long)
    # Pad on the time axis (now dim=0)
    specs_padded = pad_sequence(specs_t, batch_first=True, padding_value=0.0)
    # Restore shape to [batch, 1, n_mels, T_max]
    specs_padded = specs_padded.permute(0, 2, 1).unsqueeze(1)

    return {
        "specs": specs_padded,
        "spec_lengths": spec_lengths,
        "tokens": tokens_padded,
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
