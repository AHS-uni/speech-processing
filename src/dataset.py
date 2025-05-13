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
        - 'spec': Tensor [n_mels, T]
        - 'tokens': LongTensor [L]

    Args:
        split (str): One of {'train','val','test'} indicating which split to load.
        cfg (dict): Configuration dict with keys:
            - 'preprocessed_dir': base dir for precomputed .pt bundles
            - 'splits_dir': dir containing {split}_idx.json files
    """

    def __init__(self, split: str, cfg: dict):
        self.split = split
        self.cfg = cfg
        splits_dir = cfg["splits_dir"]
        preproc_dir = cfg["preprocessed_dir"]
        # load indices for this split
        idx_path = os.path.join(splits_dir, f"{split}_idx.json")
        with open(idx_path, "r") as f:
            self.indices = json.load(f)
            # load precomputed bundle: list of {'spec','tokens'}
        bundle_path = os.path.join(preproc_dir, f"{split}_data.pt")
        self.bundle = torch.load(bundle_path)

    def __len__(self):
        """Return number of samples in the split."""
        return len(self.bundle)

    def __getitem__(self, idx):
        """
        Fetch one sample by index.

        Args:
            idx (int): index in [0, len(self)).

        Returns:
            tuple: (spec, tokens) where
                - spec: Tensor [n_mels, T]
                - tokens: LongTensor [L]
        """
        entry = self.bundle[idx]
        spec = entry["spec"]
        tokens = entry["tokens"]
        # ensure correct dtype
        if not isinstance(tokens, torch.LongTensor):
            tokens = tokens.long()
        return spec, tokens
