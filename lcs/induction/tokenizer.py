"""Simple surface tokeniser. No ML, no external parser, no UD."""
from __future__ import annotations
import re
import struct
from dataclasses import dataclass


def _stable_node_id(lemma: str) -> int:
    """64-bit FNV-1a hash matching Rust stable_node_id and other Python callers."""
    h = 0xcbf2_9ce4_8422_2325
    for b in lemma.encode():
        h = ((h ^ b) * 0x0000_0100_0000_01b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _stable_node_id_bytes(buf: bytes) -> int:
    """64-bit FNV-1a over raw bytes. Same algorithm as _stable_node_id but takes bytes."""
    h = 0xcbf2_9ce4_8422_2325
    for b in buf:
        h = ((h ^ b) * 0x0000_0100_0000_01b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _fnv_hash(s: str) -> int:
    """32-bit FNV-1a hash for feature hashing (suffix, prefix, trigrams)."""
    h = 0x811c9dc5
    for b in s.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def contextual_node_id(
    suffix3_hash: int, prefix2_hash: int, env: int, mode: int
) -> int:
    """
    64-bit contextual FNV-1a. Python parity for Rust contextual_node_id.

    Layout (17 bytes, all little-endian):
        suffix3_hash[4 LE] ++ prefix2_hash[4 LE] ++ env[8 LE] ++ mode[1]

    mode: 0=Diamond, 1=Box, 2=Lozenge
    env:  Env bitmask (u64)
    """
    buf = (struct.pack('<I', suffix3_hash & 0xFFFFFFFF)
         + struct.pack('<I', prefix2_hash & 0xFFFFFFFF)
         + struct.pack('<Q', env & 0xFFFFFFFFFFFFFFFF)
         + bytes([mode & 0xFF]))
    return _stable_node_id_bytes(buf)


def infer_modal_mode(is_repeated: bool, normalized_position: float, is_last_token: bool) -> int:
    """
    Infer ModalMode from structural features. Mirrors Rust infer_modal_mode.
    Returns: 0=Diamond, 1=Box, 2=Lozenge
    """
    if is_repeated:
        return 1  # Box: shared/contraction
    elif normalized_position > 0.5 and not is_last_token:
        return 2  # Lozenge: displacement
    else:
        return 0  # Diamond: primary composition


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


def tokenize(text: str, language: str = "en") -> TokenSentence:
    # Strict character-level induction: the graph must organically learn morphology 
    # (morphemes/words) via structural co-occurrence, without preconceived regex biases.
    chars  = list(text)
    tokens = [Token(id=i + 1, text=c, lemma=c.lower()) for i, c in enumerate(chars)]
    return TokenSentence(tokens=tokens, language=language, text=text)


def tokenize_batch(texts: list[str], language: str = "en") -> list[TokenSentence]:
    return [tokenize(t, language) for t in texts]

