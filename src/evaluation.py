import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from jiwer import wer as compute_wer, cer as compute_cer
from typing import Callable, Dict, List, Tuple


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    ctc_loss_fn: nn.CTCLoss,
    ce_loss_fn: nn.CrossEntropyLoss,
    decode_fn: Callable[[List[int]], str],
    alpha: float,
    device: torch.device,
    checkpoint_path: str,
    show_examples: int = 5,
) -> Dict[str, float]:
    """
    Evaluate a trained HybridASR model on a dataset.

    Loads a checkpoint (optional), computes CTC, CE, combined losses,
    WER and CER over the dataset, and prints sample predictions.

    Returns a dict of metrics: {'ctc', 'ce', 'combined', 'wer', 'cer'}.
    """
    # Load weights if provided
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)

    model.eval().to(device)

    total_ctc = total_ce = total_comb = 0.0
    refs: List[str] = []
    hyps: List[str] = []
    examples: List[Tuple[str, str]] = []

    with torch.no_grad():
        for batch in loader:
            specs = batch["specs"].to(device)
            spec_lens = batch["spec_lengths"].to(device)
            tokens = batch["tokens"].to(device)
            token_lens = batch["token_lengths"].to(device)

            # Forward
            ctc_logits, dec_logits = model(specs, tokens)

            # Loss
            log_probs = torch.log_softmax(ctc_logits, dim=-1)
            input_lengths = (
                spec_lens + model.subsample_rate - 1
            ) // model.subsample_rate
            ctc_loss = ctc_loss_fn(log_probs, tokens, input_lengths, token_lens)

            B, S, C = dec_logits.shape
            ce_loss = ce_loss_fn(dec_logits.view(B * S, C), tokens.view(-1))
            combined = alpha * ctc_loss + (1 - alpha) * ce_loss

            total_ctc += ctc_loss.item()
            total_ce += ce_loss.item()
            total_comb += combined.item()

            # Decode
            preds = ctc_logits.argmax(dim=-1).transpose(0, 1)  # [B, T']
            for pred_ids, tgt_ids, tgt_len in zip(
                preds, tokens.cpu(), token_lens.cpu()
            ):
                # collapse repeats and blanks
                seq: List[int] = []
                prev = None
                for t in pred_ids.tolist():
                    if t != 0 and t != prev:
                        seq.append(t)
                        prev = t

                hyp = decode_fn(seq) if seq else ""
                ref = decode_fn(tgt_ids[:tgt_len].tolist()) if tgt_len > 0 else ""
                hyps.append(hyp)
                refs.append(ref)

                if len(examples) < show_examples:
                    examples.append((ref, hyp))

    # Metrics
    n = len(loader)
    avg_ctc = total_ctc / n
    avg_ce = total_ce / n
    avg_comb = total_comb / n
    wer_score = compute_wer(refs, hyps)
    cer_score = compute_cer(refs, hyps)

    # Display
    print("\n=== Evaluation Metrics ===")
    print(f"CTC Loss  : {avg_ctc:.4f}")
    print(f"CE Loss   : {avg_ce:.4f}")
    print(f"Combined  : {avg_comb:.4f}")
    print(f"WER       : {wer_score:.2%}")
    print(f"CER       : {cer_score:.2%}\n")

    print("=== Sample Predictions ===")
    for i, (ref, hyp) in enumerate(examples):
        print(f"[{i+1}]")
        print(f"Ref: {ref}")
        print(f"Hyp: {hyp}\n")

    return {
        "ctc": avg_ctc,
        "ce": avg_ce,
        "combined": avg_comb,
        "wer": wer_score,
        "cer": cer_score,
    }
