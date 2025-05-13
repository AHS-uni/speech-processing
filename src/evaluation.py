import torch
from torch.utils.data import DataLoader
from jiwer import wer as compute_wer, cer as compute_cer


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    ctc_loss_fn: torch.nn.CTCLoss,
    ce_loss_fn: torch.nn.CrossEntropyLoss,
    decode_fn,
    alpha: float,
    device: torch.device,
    checkpoint_path: str,
    show_examples: int = 5,
):
    # Load model checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"])

    model.eval()
    model.to(device)

    total_ctc = total_ce = total_comb = 0.0
    refs, hyps = [], []
    all_examples = []

    for batch in loader:
        specs = batch["specs"].to(device)
        spec_lens = batch["spec_lengths"].to(device)
        tokens = batch["tokens"].to(device)
        token_lens = batch["token_lengths"].to(device)

        # Forward pass
        ctc_logits, dec_logits = model(specs, tokens)

        # Compute losses
        log_probs = torch.log_softmax(ctc_logits, dim=-1)
        subsample = model.subsample_rate
        input_lengths = (spec_lens + subsample - 1) // subsample

        ctc_loss = ctc_loss_fn(log_probs, tokens, input_lengths, token_lens)
        B, S, C = dec_logits.shape
        ce_loss = ce_loss_fn(dec_logits.reshape(B * S, C), tokens.reshape(-1))
        combined_loss = alpha * ctc_loss + (1 - alpha) * ce_loss

        total_ctc += ctc_loss.item()
        total_ce += ce_loss.item()
        total_comb += combined_loss.item()

        # Decode predictions
        preds = ctc_logits.argmax(dim=-1).transpose(0, 1)  # [B, T']
        tokens = tokens.cpu()
        token_lens = token_lens.cpu()

        for i, (pred_ids, tgt_ids, tgt_len) in enumerate(
            zip(preds, tokens, token_lens)
        ):
            hyp = []
            prev = None
            for t in pred_ids.tolist():
                if t != 0 and t != prev:
                    hyp.append(t)
                    prev = t

            hyp_str = decode_fn(hyp) if hyp else ""
            ref_str = decode_fn(tgt_ids[:tgt_len].tolist()) if tgt_len > 0 else ""

            hyps.append(hyp_str)
            refs.append(ref_str)

            if len(all_examples) < show_examples:
                all_examples.append((hyp_str, ref_str))

    avg_ctc = total_ctc / len(loader)
    avg_ce = total_ce / len(loader)
    avg_comb = total_comb / len(loader)
    wer_score = compute_wer(refs, hyps)
    cer_score = compute_cer(refs, hyps)

    print("\n=== Evaluation Metrics ===")
    print(f"CTC Loss     : {avg_ctc:.4f}")
    print(f"CE Loss      : {avg_ce:.4f}")
    print(f"Combined     : {avg_comb:.4f}")
    print(f"WER          : {wer_score:.2%}")
    print(f"CER          : {cer_score:.2%}")

    print("\n=== Sample Predictions ===")
    for i, (hyp, ref) in enumerate(all_examples):
        print(f"[{i}]")
        print(f"Ref : {ref}")
        print(f"Hyp : {hyp}\n")

    return {
        "ctc": avg_ctc,
        "ce": avg_ce,
        "combined": avg_comb,
        "wer": wer_score,
        "cer": cer_score,
    }
