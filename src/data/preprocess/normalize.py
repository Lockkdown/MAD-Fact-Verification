"""Text normalization pipeline for Vietnamese fact-checking inputs."""

import unicodedata


def normalize_text(text: str, use_pyvi: bool = False) -> str:
    """Apply NFC normalization, whitespace cleanup, and optional PyVi word segmentation."""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())

    if use_pyvi:
        try:
            from pyvi import ViTokenizer
            text = ViTokenizer.tokenize(text)
        except Exception:
            pass  # fallback to unsegmented if pyvi fails

    return text.strip()
