"""
c4_sequence_extractor.py — Extract UTF-8 character sequences from mC4 for
self-supervised teacher forcing.

Streams mC4, splits each sentence into Unicode code points, and produces
TokenSequence objects. Every character is its own token. No parser, no
lemmatizer, no pretrained model.

node_id == expected_node_id == _stable_node_id(char) for every step.
This is the invariant the engine requires: the node in the pool must have
the same id as the teacher-forcing target, or ce_quality_split returns 0
on every step.

Boundary and word-level structure emerge from the training signal:
- Edge attribution weights converge on PMI (Harris successor variety).
- VDBE threshold drops at high-entropy (boundary) positions.
- GraphicaCache encodes MDL compression of stable character motifs.
- Sheaf coherence violations mark modal type discontinuities at boundaries.
- ATMS justification chains compress recurring character sequences.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def _stable_node_id(char: str) -> int:
    """
    Stable 48-bit hash of a Unicode code point used as NodeId in the ARG.

    The same character always maps to the same NodeId regardless of position,
    language, or passage. This is the vocabulary identity.
    """
    return int(hashlib.sha256(char.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFFFFFF


@dataclass
class TokenSequenceStep:
    text:             str    # the character itself
    lemma:            str    # == text (character IS its own identity)
    node_id:          int    # == expected_node_id (fixes the positional-vs-hash bug)
    expected_node_id: int    # stable hash of the character
    deprel_hash:      int    # 0 — CategoryInducer learns structure from context
    upos_hash:        int    # 0 — CategoryInducer learns structure from context


@dataclass
class TokenSequence:
    """One training example: a sentence broken into character-level steps."""
    sentence: str
    trd_id:   int
    steps:    list[TokenSequenceStep]


def _sentence_to_sequence(sentence: str, trd_id: int) -> TokenSequence:
    """
    Convert a sentence string to a TokenSequence by splitting on Unicode
    code points. node_id == expected_node_id so the engine finds the node
    in the pool and computes a meaningful CE signal.
    """
    steps = []
    for ch in sentence:  # iterates Unicode code points, not bytes
        nid = _stable_node_id(ch)
        steps.append(TokenSequenceStep(
            text=ch,
            lemma=ch,
            node_id=nid,
            expected_node_id=nid,
            deprel_hash=0,
            upos_hash=0,
        ))
    return TokenSequence(sentence=sentence, trd_id=trd_id, steps=steps)


def stream_c4_sequences(
    language:        str = "en",
    trd_assignments: Optional[dict[str, int]] = None,
    max_sentences:   int = 10_000,
) -> Iterator[TokenSequence]:
    """
    Stream self-supervised teacher-forcing sequences from mC4.

    Each Unicode code point provides its own label via _stable_node_id(char).
    No external supervision. No parser. No model.
    """
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "induction"))
        from mc4_stream import stream_mc4, _split_sentences
    except ImportError:
        logger.warning("mc4_stream not available; yielding empty sequence stream")
        return

    count = 0
    for item in stream_mc4(language=language):
        if count >= max_sentences:
            break
        text = item.get("text", "")
        if not text.strip():
            continue
        for sentence in _split_sentences(text):
            if count >= max_sentences:
                break
            if not sentence.strip():
                continue
            trd_id = 0
            if trd_assignments:
                trd_id = trd_assignments.get(sentence[:32], 0)
            yield _sentence_to_sequence(sentence, trd_id)
            count += 1
