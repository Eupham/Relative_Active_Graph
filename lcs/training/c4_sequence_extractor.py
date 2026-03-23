"""Extract token sequences from mC4 for teacher forcing. No UD, no Stanza."""
from __future__ import annotations
import hashlib, logging, re
from dataclasses import dataclass
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_SPLIT_RE = re.compile(r"\w+(?:'\w+)*|[^\w\s]", re.UNICODE)

def _stable_node_id(lemma: str) -> int:
    return int(hashlib.sha256(lemma.encode()).hexdigest(), 16) & 0xFFFFFFFFFFFF

def _fnv_hash(s: str) -> int:
    h = 0x811c9dc5
    for b in s.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


@dataclass
class TokenSequenceStep:
    text:             str
    lemma:            str
    expected_node_id: int
    node_id:          int
    suffix3_hash:     int
    prefix2_hash:     int


@dataclass
class TokenSequence:
    sentence: str
    trd_id:   int
    steps:    list[TokenSequenceStep]


def extract_sequence(text: str, trd_id: int = 0) -> TokenSequence:
    words = _SPLIT_RE.findall(text)
    steps = []
    for i, w in enumerate(words):
        lemma = w.lower()
        steps.append(TokenSequenceStep(
            text=w, lemma=lemma,
            expected_node_id=_stable_node_id(lemma),
            node_id=i + 1,
            suffix3_hash=_fnv_hash(lemma[-3:]) if len(lemma) >= 3 else _fnv_hash(lemma),
            prefix2_hash=_fnv_hash(lemma[:2])  if len(lemma) >= 2 else _fnv_hash(lemma),
        ))
    return TokenSequence(sentence=text, trd_id=trd_id, steps=steps)


def stream_c4_sequences(
    language:        str             = "en",
    trd_assignments: Optional[dict]  = None,
    max_sentences:   int             = 10_000,
    use_stanza:      bool            = False,
) -> Iterator[TokenSequence]:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "induction"))
    from mc4_stream import stream_mc4
    count = 0
    for item in stream_mc4(language, max_samples=max_sentences):
        text = item.get("text", "").strip()
        if not text:
            continue
        trd_id = trd_assignments.get(text, 0) if trd_assignments else 0
        seq    = extract_sequence(text, trd_id)
        if seq.steps:
            yield seq
            count += 1
            if count >= max_sentences:
                break
