"""
c4_sequence_extractor.py — Extract token sequences from mC4 for teacher forcing.

Streams the mC4 dataset, runs UD parsing, and produces (sentence, ud_tree) pairs
suitable for the sequential trainer. Integrates with the existing mc4_stream.py.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ── UD stub types (used when stanza is unavailable) ──────────────────────────

@dataclass
class StubToken:
    id: int
    text: str
    lemma: str
    upos: str
    head: int
    deprel: str
    feats: dict[str, str] = field(default_factory=dict)


@dataclass
class StubTree:
    tokens: list[StubToken]
    language: str
    text: str

    def token_by_id(self, tid: int) -> Optional[StubToken]:
        return next((t for t in self.tokens if t.id == tid), None)


# ── UD parsing ────────────────────────────────────────────────────────────────

def parse_ud_trees(
    sentences: list[str],
    language: str = "en",
    use_stanza: bool = True,
) -> list[Any]:
    """
    Parse sentences into UD trees.

    Tries stanza first; falls back to stub single-token trees when unavailable.
    """
    if use_stanza:
        try:
            return _parse_with_stanza(sentences, language)
        except Exception as exc:
            logger.warning("stanza unavailable (%s), using stub trees", exc)

    return [_stub_tree(s, language) for s in sentences]


def _parse_with_stanza(sentences: list[str], language: str) -> list[Any]:
    import stanza
    nlp = stanza.Pipeline(lang=language, processors="tokenize,pos,lemma,depparse")
    trees = []
    for sent in sentences:
        doc = nlp(sent)
        for sentence in doc.sentences:
            tokens = []
            for word in sentence.words:
                tokens.append(StubToken(
                    id=word.id,
                    text=word.text,
                    lemma=word.lemma or word.text,
                    upos=word.upos or "X",
                    head=word.head,
                    deprel=word.deprel or "dep",
                    feats={},
                ))
            trees.append(StubTree(tokens=tokens, language=language, text=sentence.text))
    return trees


def _stub_tree(text: str, language: str) -> StubTree:
    """Create a single-root stub tree for a sentence."""
    words = text.strip().split()
    tokens = [
        StubToken(
            id=i + 1,
            text=w,
            lemma=w.lower(),
            upos="X",
            head=0 if i == 0 else 1,
            deprel="root" if i == 0 else "dep",
        )
        for i, w in enumerate(words)
    ]
    return StubTree(tokens=tokens, language=language, text=text)


# ── Sequence extraction ───────────────────────────────────────────────────────

@dataclass
class TokenSequence:
    """One training example: a sentence broken into teacher-forcing steps."""
    sentence: str
    trd_id:   int
    steps: list["TokenSequenceStep"]


@dataclass
class TokenSequenceStep:
    text:             str
    lemma:            str
    expected_node_id: int   # stable hash of lemma (replaces expected_edge_id)
    node_id:          int
    deprel_hash:      int   # hash of raw deprel string — no category assignment
    upos_hash:        int   # hash of UPOS tag — no hard mapping


def _stable_node_id(lemma: str) -> int:
    """Stable 48-bit hash of lemma used as NodeId in the ARG."""
    return int(hashlib.sha256(lemma.encode()).hexdigest(), 16) & 0xFFFFFFFFFFFF


def _hash_str(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) & 0xFFFF


def extract_sequence(
    tree: Any,
    trd_id: int,
    base_edge_id: int = 1,
) -> TokenSequence:
    """
    Convert a UD tree into a `TokenSequence` for teacher forcing.

    Each token becomes one step. The expected_node_id is a stable hash of the
    token's lemma. No hard category labels are assigned — structural hashes
    (deprel_hash, upos_hash) are provided for post-hoc clustering by the
    CategoryInducer in the Rust engine.
    """
    steps = []
    for tok in getattr(tree, "tokens", []):
        lemma = getattr(tok, "lemma", tok.text.lower())
        steps.append(TokenSequenceStep(
            text=tok.text,
            lemma=lemma,
            expected_node_id=_stable_node_id(lemma),
            node_id=tok.id,
            deprel_hash=_hash_str(getattr(tok, "deprel", "dep")),
            upos_hash=_hash_str(getattr(tok, "upos", "X")),
        ))
    return TokenSequence(
        sentence=getattr(tree, "text", ""),
        trd_id=trd_id,
        steps=steps,
    )


def stream_c4_sequences(
    language: str = "en",
    trd_assignments: Optional[dict[str, int]] = None,
    max_sentences: int = 10_000,
) -> Iterator[TokenSequence]:
    """
    Stream token sequences from mC4 via mc4_stream.py.

    `trd_assignments`: maps sentence hash → TRD ID (optional; defaults to 0).
    """
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "induction"))
        from mc4_stream import stream_mc4
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
        for sentence_text in _split_sentences(text):
            if count >= max_sentences:
                break
            tree = _stub_tree(sentence_text, language)
            trd_id = 0
            if trd_assignments:
                key = sentence_text[:32]
                trd_id = trd_assignments.get(key, 0)
            seq = extract_sequence(tree, trd_id=trd_id)
            yield seq
            count += 1


def _split_sentences(text: str, max_len: int = 200) -> list[str]:
    """Naive sentence splitter (no stanza dependency)."""
    import re
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip() and len(s) <= max_len]
