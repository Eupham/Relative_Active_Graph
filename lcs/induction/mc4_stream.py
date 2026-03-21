"""Stream sentences from the mC4 multilingual corpus (allenai/c4, mC4 variant).

Yields dicts: {"text": str, "language": str, "url": str}.
Supports 108 languages. Streaming mode avoids downloading the full dataset.
"""
from __future__ import annotations

import logging
from typing import Iterator

logger = logging.getLogger(__name__)

# Languages in the mC4 corpus (representative subset; full list has 108).
MC4_LANGUAGES = [
    "af", "am", "ar", "az", "be", "bg", "bn", "ca", "cs", "cy",
    "da", "de", "el", "en", "eo", "es", "et", "eu", "fa", "fi",
    "fr", "fy", "ga", "gl", "gu", "ha", "hi", "hr", "ht", "hu",
    "hy", "id", "ig", "is", "it", "iw", "ja", "ka", "kk", "km",
    "kn", "ko", "ku", "ky", "la", "lb", "lo", "lt", "lv", "mg",
    "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl",
    "no", "ny", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si",
    "sk", "sl", "sm", "sn", "so", "sq", "sr", "st", "su", "sv",
    "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "ug",
    "uk", "ur", "uz", "vi", "xh", "yi", "yo", "zh", "zu",
]

# Languages requiring FST morphological decomposition before type assignment.
AGGLUTINATIVE_LANGUAGES = {"fi", "tr", "hu", "ka", "sw", "tl", "az", "kk", "ky", "uz"}
POLYSYNTHETIC_LANGUAGES  = {"my"}  # Burmese approximation; full list includes Nahuatl etc.
FST_REQUIRED_LANGUAGES   = AGGLUTINATIVE_LANGUAGES | POLYSYNTHETIC_LANGUAGES


def stream_mc4(
    language:    str,
    max_samples: int = 10_000,
    split:       str = "train",
) -> Iterator[dict]:
    """Stream sentences from mC4 for a given language.

    Args:
        language:    ISO 639-1 code (must be in MC4_LANGUAGES).
        max_samples: Maximum sentences to yield (streaming).
        split:       Dataset split ("train" or "validation").

    Yields:
        {"text": str, "language": str, "url": str}
    """
    if language not in MC4_LANGUAGES:
        raise ValueError(f"Language '{language}' not in mC4. Supported: {MC4_LANGUAGES}")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets' package: pip install datasets")

    logger.info("Streaming mC4 for language=%s, split=%s, max=%d", language, split, max_samples)

    dataset = load_dataset(
        "allenai/c4",
        name=language,
        split=split,
        streaming=True,

    )

    count = 0
    for example in dataset:
        if count >= max_samples:
            break
        text = example.get("text", "").strip()
        if not text:
            continue
        # Split into sentences (simple heuristic; Stanza handles proper segmentation).
        for sent in _split_sentences(text):
            if count >= max_samples:
                break
            yield {"text": sent, "language": language, "url": example.get("url", "")}
            count += 1

    logger.info("Streamed %d sentences for language=%s", count, language)


def _split_sentences(text: str, max_length: int = 512) -> list[str]:
    """Basic sentence splitting: split on '. ', '! ', '? ' boundaries."""
    import re
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s[:max_length] for s in sents if len(s.split()) >= 3]


def requires_fst(language: str) -> bool:
    """Does this language require FST decomposition before type assignment?"""
    return language in FST_REQUIRED_LANGUAGES


def stream_multilingual(
    languages:   list[str] | None = None,
    max_per_lang: int = 1_000,
) -> Iterator[dict]:
    """Stream sentences interleaved across multiple languages."""
    langs = languages or MC4_LANGUAGES[:10]  # default: first 10 for dev
    for lang in langs:
        try:
            yield from stream_mc4(lang, max_samples=max_per_lang)
        except Exception as exc:
            logger.warning("Skipping language %s: %s", lang, exc)


if __name__ == "__main__":
    import sys
    import json
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    for item in stream_mc4(lang, max_samples=5):
        print(json.dumps(item, ensure_ascii=False))
