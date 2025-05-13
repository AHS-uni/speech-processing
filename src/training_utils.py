"""
Training utilities for HybridASR: loss computation and epoch loops.

This module provides:
- compute_batch_loss: Compute CTC, Cross-Entropy, and combined loss for a batch.
- run_epoch: Run one epoch of training or validation, returning losses + WER/CER.
"""

from typing import Callable, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from jiwer import wer as compute_wer, cer as compute_cer


def compute_batch_loss(
    batch: dict,
    model: nn.Module,
    ctc_loss_fn: nn.CTCLoss,
    ce_loss_fn: nn.CrossEntropyLoss,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute CTC, cross-entropy, and combined loss for a single batch, and return logits.

    Args:
        batch (dict): A batch dict with:
            - 'specs': Tensor of shape [B, 1, n_mels, T]
            - 'spec_lengths': LongTensor of shape [B]
            - 'tokens': LongTensor of shape [B, S]
            - 'token_lengths': LongTensor of shape [B]
        model (nn.Module): HybridASR model instance.
        ctc_loss_fn (nn.CTCLoss): CTC loss function.
        ce_loss_fn (nn.CrossEntropyLoss): Cross-entropy loss function.
        alpha (float): Weight for CTC loss in combined loss.
        device (torch.device): Computation device.

    Returns:
        ctc_loss (Tensor): Scalar CTC loss.
        ce_loss (Tensor): Scalar cross-entropy loss.
        combined_loss (Tensor): Scalar weighted sum of losses.
        ctc_logits (Tensor): Logits for CTC head, shape [T', B, C].
        dec_logits (Tensor): Logits for decoder head, shape [B, S, C].
    """
    # Move inputs to device
    specs = batch["specs"].to(device)  # [B, 1, n_mels, T]
    spec_lens = batch["spec_lengths"].to(device)  # [B]
    tokens = batch["tokens"].to(device)  # [B, S]
    token_lens = batch["token_lengths"].to(device)  # [B]

    # Forward pass
    ctc_logits, dec_logits = model(specs, tokens)
    # ctc_logits: [T', B, C], dec_logits: [B, S, C]

    # CTC loss uses downsampled input lengths
    # Get log-probs
    log_probs = torch.log_softmax(ctc_logits, dim=-1)  # [T', B, C]
    # Compute downsamples lengths: ceil(T_i / subsample_rate)
    subsample = model.subsample_rate  # == 2
    input_lengths = ((spec_lens + subsample - 1) // subsample).to(device)  # [B]
    # Call CTC loss with padded tokens shape [B, S]
    ctc_loss = ctc_loss_fn(
        log_probs, tokens, input_lengths=input_lengths, target_lengths=token_lens
    )

    # Cross-Entropy loss over decoder outputs
    B, S, C = dec_logits.shape
    ce_loss = ce_loss_fn(dec_logits.reshape(B * S, C), tokens.reshape(-1))

    # Combined loss
    combined_loss = alpha * ctc_loss + (1.0 - alpha) * ce_loss

    return ctc_loss, ce_loss, combined_loss, ctc_logits, dec_logits


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    ctc_loss_fn: nn.CTCLoss,
    ce_loss_fn: nn.CrossEntropyLoss,
    decode_fn: Callable[[List[int]], str],
    alpha: float,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = True,
) -> tuple[float, float, float, float, float]:
    """
    Run one epoch of training or evaluation.

    Args:
        model: HybridASR model.
        loader: DataLoader providing batches.
        ctc_loss_fn: CTC loss function.
        ce_loss_fn: Cross-entropy loss function.
        decode_fn: Callable to decode predicted token IDs into text.
        alpha: Weight of CTC loss in combined loss.
        device: Computation device (e.g., 'cuda').
        optimizer: Optimizer to update model (required if train=True).
        train: Whether to run in training mode.

    Returns:
        Average CTC loss, CE loss, combined loss, WER, CER for the epoch.
    """
    if train and optimizer is None:
        raise ValueError("Optimizer required in training mode")

    model.train() if train else model.eval()

    total_ctc = total_ce = total_combined = 0.0
    refs, hyps = [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            # Ensure required keys are present
            if not all(
                k in batch for k in ("specs", "spec_lengths", "tokens", "token_lengths")
            ):
                continue  # skip incomplete batches

            # Forward + loss
            ctc_loss, ce_loss, combined_loss, ctc_logits, _ = compute_batch_loss(
                batch, model, ctc_loss_fn, ce_loss_fn, alpha, device
            )

            # Backpropagation
            if train:
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

            # Accumulate losses
            total_ctc += ctc_loss.item()
            total_ce += ce_loss.item()
            total_combined += combined_loss.item()

            # Decode predictions
            preds = ctc_logits.argmax(dim=-1).transpose(0, 1)  # [B, T']
            targets = batch["tokens"].cpu()
            target_lens = batch["token_lengths"].cpu()

            for pred_ids, tgt_ids, tgt_len in zip(preds, targets, target_lens):
                if tgt_len == 0:
                    continue  # Skip invalid example

                hyp = []
                prev = None
                for t in pred_ids.tolist():
                    if t != 0 and t != prev:
                        hyp.append(t)
                        prev = t

                hyp_str = decode_fn(hyp)
                ref_str = decode_fn(tgt_ids[:tgt_len].tolist())

                hyps.append(hyp_str)
                refs.append(ref_str)

    num_batches = len(loader)
    avg_ctc = total_ctc / num_batches
    avg_ce = total_ce / num_batches
    avg_combined = total_combined / num_batches
    wer_score = compute_wer(refs, hyps)
    cer_score = compute_cer(refs, hyps)

    return avg_ctc, avg_ce, avg_combined, wer_score, cer_score
