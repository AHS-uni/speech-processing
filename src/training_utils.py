"""
Training utilities for HybridASR: loss computation and epoch loops.

This module provides:
- compute_batch_loss: Compute CTC, Cross-Entropy, and combined loss for a batch.
- run_epoch: Run one epoch of training or validation, returning losses + WER/CER.
"""

from typing import Callable, Optional, List, Tuple, Any
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute CTC, cross-entropy, and combined loss for a single batch, returning:
      (ctc_loss, ce_loss, combined_loss, ctc_logits, dec_logits)

    Uses teacher-forcing with an SOS token, shifting decoder inputs by one.
    """
    # Move everything to device
    specs = batch["specs"].to(device)  # [B, 1, n_mels, T]
    spec_lens = batch["spec_lengths"].to(device)  # [B]
    tokens = batch["tokens"].to(device)  # [B, S]
    token_lens = batch["token_lengths"].to(device)  # [B]

    # CTC branch uses full tokens sequence
    ctc_logits, _ = model(specs, tokens)  # ctc_logits: [T', B, C]

    # CTC loss
    log_probs = torch.log_softmax(ctc_logits, dim=-1)  # [T', B, C]
    subsample = model.subsample_rate
    input_lens = ((spec_lens + subsample - 1) // subsample).to(device)
    ctc_loss = ctc_loss_fn(
        log_probs, tokens, input_lengths=input_lens, target_lengths=token_lens
    )

    # Prepare shifted inputs for decoder
    # Assume tokens[b,0] is SOS, tokens[b,1:] are actual tokens, possibly padded
    decoder_input = tokens[:, :-1]  # [B, S-1]
    decoder_target = tokens[:, 1:]  # [B, S-1]
    dec_lens = torch.clamp(token_lens - 1, min=0)

    # Decoder branch
    _, dec_logits = model(specs, decoder_input)  # dec_logits: [B, S-1, C]

    # CE loss over decoder outputs
    B, S1, C = dec_logits.shape
    ce_loss = ce_loss_fn(dec_logits.reshape(B * S1, C), decoder_target.reshape(-1))

    # Combined
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
) -> Tuple[float, float, float, float, float]:
    """
    Run one epoch of training or evaluation, using the decoder head for metrics.

    Args:
        model: ASR model.
        loader: DataLoader for data.
        ctc_loss_fn: CTC loss function.
        ce_loss_fn: CrossEntropy loss function.
        decode_fn: function mapping token ID list to string.
        alpha: CTC weight.
        device: computation device.
        optimizer: optimizer for training (required if train=True).
        train: True for training, False for eval.

    Returns:
        Tuple of (avg_ctc_loss, avg_ce_loss, avg_combined_loss, wer, cer).
    """
    if train and optimizer is None:
        raise ValueError("Optimizer required in training mode.")

    model.train() if train else model.eval()

    total_ctc = total_ce = total_comb = 0.0
    refs, hyps = [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            # Move batch to device
            specs = batch["specs"].to(device)
            tokens = batch["tokens"].to(device)
            spec_lens = batch["spec_lengths"].to(device)
            token_lens = batch["token_lengths"].to(device)

            if train:
                optimizer.zero_grad()

            # Compute losses and logits
            ctc_loss, ce_loss, combined_loss, ctc_logits, dec_logits = (
                compute_batch_loss(
                    {
                        "specs": specs,
                        "spec_lengths": spec_lens,
                        "tokens": tokens,
                        "token_lengths": token_lens,
                    },
                    model,
                    ctc_loss_fn,
                    ce_loss_fn,
                    alpha,
                    device,
                )
            )

            if train:
                combined_loss.backward()
                optimizer.step()

            total_ctc += ctc_loss.item()
            total_ce += ce_loss.item()
            total_comb += combined_loss.item()

            # Decode using decoder head (greedy)
            # dec_logits: [B, S-1, C]
            dec_ids = dec_logits.argmax(dim=-1).cpu().tolist()
            B = len(dec_ids)

            for b in range(B):
                # Reference: skip SOS (first token)
                tgt_len = token_lens[b].item() - 1
                ref_ids = tokens[b].cpu().tolist()[1 : 1 + tgt_len]
                ref_str = decode_fn(ref_ids)
                if not ref_str:
                    continue

                # Hypothesis: strip padding (0)
                hyp_ids = [tok for tok in dec_ids[b] if tok != 0]
                hyp_str = decode_fn(hyp_ids)

                refs.append(ref_str)
                hyps.append(hyp_str)

    num_batches = len(loader)
    avg_ctc = total_ctc / num_batches
    avg_ce = total_ce / num_batches
    avg_comb = total_comb / num_batches
    wer_score = compute_wer(refs, hyps)
    cer_score = compute_cer(refs, hyps)
    return avg_ctc, avg_ce, avg_comb, wer_score, cer_score
