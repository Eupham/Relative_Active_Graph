"""FST-based morphological decomposition for agglutinative/polysynthetic languages.

Required before MTLG type assignment for languages in FST_REQUIRED_LANGUAGES.
A single word in Turkish/Finnish can encode a full clause; FST decomposes it
into morpheme sequences that MTLG can then assign types to.

Uses sentencepiece for analytic/fusional languages (fallback).
Uses morphological rules (simplified FST) for agglutinative languages.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from mc4_stream import FST_REQUIRED_LANGUAGES

logger = logging.getLogger(__name__)


@dataclass
class MorphemeDecomposition:
    """Result of FST decomposition: the word split into typed morphemes."""
    original:   str
    language:   str
    morphemes:  list[str]         # individual morphemes in order
    tags:       list[str]         # morphological tags (STEM, TENSE, PERSON, etc.)
    is_clause_level: bool         # True if this word encodes a full clause


class SimplifiedFst:
    """A simplified Finite-State Transducer for morphological decomposition.

    In production, this would be backed by a full FST library (e.g., rustfst via
    Python bindings, or hfst). Here we implement a rule-based approximation
    sufficient for the CSRRE LCS pipeline.
    """

    # Turkish morpheme boundary patterns (simplified).
    TURKISH_SUFFIXES = [
        (r"(yor)$",    "PROG"),  # progressive
        (r"(di)$",     "PAST"),  # past tense
        (r"(mek)$",    "INF"),   # infinitive
        (r"(ler|lar)$", "PL"),   # plural
        (r"(im|ım|üm|um)$", "1SG"),  # 1st sg
        (r"(sin|sın|sün|sun)$", "2SG"),
        (r"(iz|ız|üz|uz)$", "1PL"),
        (r"(da|de|ta|te)$", "LOC"),
        (r"(dan|den|tan|ten)$", "ABL"),
        (r"(a|e|ya|ye)$", "DAT"),
        (r"(ı|i|u|ü|yı|yi|yu|yü)$", "ACC"),
        (r"(ın|in|un|ün|nın|nin|nun|nün)$", "GEN"),
    ]

    FINNISH_SUFFIXES = [
        (r"(ssa|ssä)$", "INESS"),   # inessive
        (r"(sta|stä)$", "ELAT"),    # elative
        (r"(lle)$",     "ALLAT"),   # allative
        (r"(lla|llä)$", "ADESS"),   # adessive
        (r"(lta|ltä)$", "ABLAT"),   # ablative
        (r"(n)$",       "GEN"),     # genitive
        (r"(t)$",       "PL_NOM"),  # plural nominative
        (r"(ksi)$",     "TRANSL"),  # translative
        (r"(tta|ttä)$", "ABESS"),   # abessive
    ]

    def __init__(self, language: str):
        self.language = language
        self.suffix_rules = self._load_suffix_rules(language)

    def _load_suffix_rules(self, language: str) -> list[tuple[str, str]]:
        if language == "tr":
            return self.TURKISH_SUFFIXES
        elif language == "fi":
            return self.FINNISH_SUFFIXES
        else:
            return []  # other languages: BPE fallback

    def decompose(self, word: str) -> MorphemeDecomposition:
        """Decompose a word into morphemes by iteratively stripping suffixes."""
        remaining = word.lower()
        morphemes = []
        tags = []
        is_clause = False

        for pattern, tag in self.suffix_rules:
            m = re.search(pattern, remaining)
            if m:
                suffix   = m.group(1)
                stem     = remaining[:m.start()]
                remaining = stem
                morphemes.insert(0, suffix)
                tags.insert(0, tag)
                if tag in ("PROG", "PAST", "INF"):
                    is_clause = True

        morphemes.insert(0, remaining)  # stem is first
        tags.insert(0, "STEM")

        return MorphemeDecomposition(
            original=word,
            language=self.language,
            morphemes=morphemes,
            tags=tags,
            is_clause_level=is_clause,
        )

    def decompose_sentence(self, sentence: str) -> list[MorphemeDecomposition]:
        return [self.decompose(w) for w in sentence.split()]


def get_decomposer(language: str) -> Optional["SimplifiedFst"]:
    """Get a decomposer for `language`, or None if not needed (use BPE instead)."""
    if language in FST_REQUIRED_LANGUAGES:
        return SimplifiedFst(language)
    return None


def preprocess_for_type_assignment(sentence: str, language: str) -> list[str]:
    """Convert a sentence to a token sequence suitable for MTLG type assignment.

    For agglutinative/polysynthetic: FST decompose → morpheme sequence.
    For analytic/fusional: sentencepiece BPE tokenization.
    """
    if language in FST_REQUIRED_LANGUAGES:
        fst = SimplifiedFst(language)
        decomps = fst.decompose_sentence(sentence)
        # Flatten to morpheme sequence: stem + each morpheme tag as a pseudo-token.
        tokens = []
        for d in decomps:
            tokens.extend([f"{m}[{t}]" for m, t in zip(d.morphemes, d.tags)])
        return tokens
    else:
        # Analytic/fusional: simple whitespace tokenization (BPE would require sentencepiece model).
        return sentence.split()


if __name__ == "__main__":
    import json
    examples = [
        ("tr", "gidiyorum"),       # Turkish: "I am going"
        ("fi", "talossani"),       # Finnish: "in my house"
        ("en", "the cat sat"),     # English: no FST needed
    ]
    for lang, word in examples:
        decomposer = get_decomposer(lang)
        if decomposer:
            d = decomposer.decompose(word)
            print(json.dumps({"lang": lang, "word": word,
                              "morphemes": d.morphemes, "tags": d.tags,
                              "is_clause": d.is_clause_level}))
        else:
            tokens = preprocess_for_type_assignment(word, lang)
            print(json.dumps({"lang": lang, "word": word, "tokens": tokens, "fst": False}))
