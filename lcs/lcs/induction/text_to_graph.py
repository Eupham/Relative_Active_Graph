"""
text_to_graph.py — Direct text → sequential MtlgGraph.
No UD, no Stanza. All edges are Diamond (◇).
Categories are 0 (DEFAULT) until the inducer fits.
"""
from __future__ import annotations
from dataclasses import dataclass
from boundary_inducer import BoundaryInducer, ParseLeaf


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


def sentence_to_mtlg(leaves: list[ParseLeaf], language: str) -> MtlgGraph:
    nodes = [MtlgNode(token_id=leaf.id, text=leaf.text, lemma=leaf.lemma) for leaf in leaves]
    edges = [
        MtlgEdge(src_id=leaves[i].id, dst_id=leaves[i + 1].id)
        for i in range(len(leaves) - 1)
    ]
    return MtlgGraph(nodes=nodes, edges=edges, language=language)


def text_to_mtlg(text: str, language: str = "en") -> MtlgGraph:
    leaves = BoundaryInducer().parse(text)
    return sentence_to_mtlg(leaves, language)
