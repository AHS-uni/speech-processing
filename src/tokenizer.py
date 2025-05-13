"""
SentencePiece tokenizer trainer and loader.
"""

import sentencepiece as spm


def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 1000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    num_threads: int = 8,
):
    """Train a SentencePiece tokenizer and write model & vocab files.

    Args:
        input_file (str): Path to newline-delimited cleaned transcripts.
        model_prefix (str): Path prefix for output files
            (`<prefix>.model` and `<prefix>.vocab`).
        vocab_size (int): Number of subword tokens.
        model_type (str): One of {"unigram","bpe","word","char"}.
        character_coverage (float): Portion of characters covered.
        num_threads (int): Parallel threads for training.

    Returns:
        Tuple[str, str]: Filenames of the trained model and vocab.
    """
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
    )
    return f"{model_prefix}.model", f"{model_prefix}.vocab"


def load_tokenizer(model_file: str) -> spm.SentencePieceProcessor:
    """Load a trained SentencePiece model.

    Args:
        model_file (str): Path to `<model_prefix>.model`.

    Returns:
        SentencePieceProcessor: Ready for `encode`/`decode`.
    """
    return spm.SentencePieceProcessor(model_file=model_file)
