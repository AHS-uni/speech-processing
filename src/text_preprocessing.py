"""
Text normalization utilities.
"""

import re
import unicodedata

# Pre-compiled regex patterns
_CURLY_APOSTROPHE = re.compile(r"[\u2018\u2019]")  # curly quotes
_MULTI_HYPHEN = re.compile(r"-{2,}")  # multiple hyphens
_MULTI_SPACE = re.compile(r"\s{2,}")  # extra whitespace
_FORBIDDEN = re.compile(r"[^a-z0-9!\'(),\-.:;? ]")  # invalid chars


def normalize_text(text: str) -> str:
    """Clean and normalize a transcript string.

    Steps:
        1. Unicode normalization (NFD) → strip combining diacritics
        2. Replace curly apostrophes → ASCII apostrophe
        3. Collapse multi-hyphens to single '-'
        4. Lowercase the text
        5. Remove any characters outside [a–z0–9!'(),-.:;? ] by replacing with space
        6. Collapse runs of whitespace to single spaces, trim ends

    Args:
        text (str): The raw or “normalized_transcript” string from LJSpeech.

    Returns:
        str: The fully cleaned, lowercase string.
    """
    # Unicode decomposition + remove diacritics
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # Punctuation and casing fixes
    text = _CURLY_APOSTROPHE.sub("'", text)
    text = _MULTI_HYPHEN.sub("-", text)
    text = text.lower()
    # Filter forbidden characters
    text = _FORBIDDEN.sub(" ", text)
    # Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text
