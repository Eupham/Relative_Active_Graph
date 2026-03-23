"""Simple surface tokeniser. No ML, no external parser, no UD."""
from __future__ import annotations
import re
from dataclasses import dataclass


def _stable_node_id(lemma: str) -> int:
    """64-bit FNV-1a hash matching Rust stable_node_id and other Python callers."""
    h = 0xcbf2_9ce4_8422_2325
    for b in lemma.encode():
        h = ((h ^ b) * 0x0000_0100_0000_01b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _fnv_hash(s: str) -> int:
    """32-bit FNV-1a hash for feature hashing (suffix, prefix, trigrams)."""
    h = 0x811c9dc5
    for b in s.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


@dataclass
class Token:
    id:    int
    text:  str
    lemma: str


@dataclass
class TokenSentence:
    tokens:   list[Token]
    language: str
    text:     str

    def token_by_id(self, tid: int) -> Token | None:
        return next((t for t in self.tokens if t.id == tid), None)


_SPLIT_RE = re.compile(r"\w+(?:'\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text: str, language: str = "en") -> TokenSentence:
    words  = _SPLIT_RE.findall(text)
    tokens = [Token(id=i + 1, text=w, lemma=w.lower()) for i, w in enumerate(words)]
    return TokenSentence(tokens=tokens, language=language, text=text)


def tokenize_batch(texts: list[str], language: str = "en") -> list[TokenSentence]:
    return [tokenize(t, language) for t in texts]
