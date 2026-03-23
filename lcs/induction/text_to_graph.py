"""
text_to_graph.py — Direct text → sequential MtlgGraph.
No UD, no Stanza. All edges are Diamond (◇).
Categories are 0 (DEFAULT) until the inducer fits.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from tokenizer import Token, TokenSentence, tokenize, _fnv_hash, _stable_node_id


@dataclass
class MtlgEdge:
    src_id:      int
    dst_id:      int
    modal_mode:  str   = "diamond"
    category_id: int   = 0
    arity:       int   = 0


@dataclass
class MtlgNode:
    token_id:    int
    text:        str
    lemma:       str
    modal_mode:  str   = "diamond"
    category_id: int   = 0
    arity:       int   = 0


@dataclass
class MtlgGraph:
    nodes:    list[MtlgNode]
    edges:    list[MtlgEdge]
    language: str


def sentence_to_mtlg(sentence: TokenSentence) -> MtlgGraph:
    nodes = [MtlgNode(token_id=t.id, text=t.text, lemma=t.lemma) for t in sentence.tokens]
    edges = [
        MtlgEdge(src_id=sentence.tokens[i].id, dst_id=sentence.tokens[i + 1].id)
        for i in range(len(sentence.tokens) - 1)
    ]
    return MtlgGraph(nodes=nodes, edges=edges, language=sentence.language)


def text_to_mtlg(text: str, language: str = "en") -> MtlgGraph:
    return sentence_to_mtlg(tokenize(text, language))
