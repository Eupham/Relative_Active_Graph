"""Convert UD dependency trees to MTLG modal edge assignments.

Primary dependencies → ◇ (Diamond) mode edges (linear resource use, tree-forming).
Shared arguments (reentrancy in enhanced UD) → □ (Box) mode edges (contraction, DAG).
Long-range / extracted dependencies → ◊ (Lozenge) mode edges (discontinuous).

UCCA categories are assigned based on UD UPOS and deprel:
  - Process/Event: VERB with verbal deprel (root, csubj, ccomp, xcomp, advcl)
  - Participant:   NOUN/PROPN with core dep (nsubj, obj, iobj)
  - State:         ADJ/AUX with predicative function
  - Scene:         clause-level construct (coordinating subgraphs)
  - Adverbial:     advmod, obl
  - Connector:     cc, mark, punct
  - Ground:        discourse, vocative
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ud_parser import UdToken, UdTree, UD_CORE_DEPS, UD_NONCORE_DEPS, REENTRANT_RELATIONS

ModalModeStr = str  # "diamond" | "box" | "lozenge"

# Category IDs (mirror Rust TypeCategory constants):
#   0 = DEFAULT/unassigned, 1 = Process, 2 = Connector, 3 = Ground,
#   4 = Adverbial, 5 = State, 6 = Participant
CATEGORY_DEFAULT     = 0
CATEGORY_PROCESS     = 1
CATEGORY_CONNECTOR   = 2
CATEGORY_GROUND      = 3
CATEGORY_ADVERBIAL   = 4
CATEGORY_STATE       = 5
CATEGORY_PARTICIPANT = 6


@dataclass
class MtlgEdge:
    src_id:      int
    dst_id:      int
    deprel:      str
    modal_mode:  ModalModeStr
    category_id: int           # TypeCategory numeric ID
    arity:       int           # remaining args for functor type (0 = saturated)


@dataclass
class MtlgNode:
    token_id:    int
    text:        str
    lemma:       str
    upos:        str
    modal_mode:  ModalModeStr
    category_id: int
    arity:       int


@dataclass
class MtlgGraph:
    nodes: list[MtlgNode]
    edges: list[MtlgEdge]
    language: str


def assign_ucca_category(tok: UdToken) -> int:
    """Assign UCCA category ID from UPOS and deprel (structural evidence only)."""
    upos   = tok.upos.upper()
    deprel = tok.deprel.lower()
    if upos == "VERB" or deprel in ("root", "ccomp", "xcomp", "advcl", "csubj"):
        return CATEGORY_PROCESS
    if upos in ("NOUN", "PROPN", "PRON") and deprel in UD_CORE_DEPS:
        return CATEGORY_PARTICIPANT
    if upos in ("ADJ", "AUX") and deprel in ("amod", "cop", "aux"):
        return CATEGORY_STATE
    if deprel in ("advmod", "obl"):
        return CATEGORY_ADVERBIAL
    if deprel in ("cc", "mark", "punct"):
        return CATEGORY_CONNECTOR
    if deprel in ("discourse", "vocative"):
        return CATEGORY_GROUND
    return CATEGORY_DEFAULT


def assign_modal_mode(tok: UdToken, deprel: str) -> ModalModeStr:
    """Assign modal mode from deprel and enhanced UD features."""
    if tok.is_reentrant or any(r in deprel for r in REENTRANT_RELATIONS):
        return "box"       # □ — shared argument (contraction)
    if deprel in ("acl:relcl", "nsubj:outer", "obj:outer"):
        return "lozenge"   # ◊ — long-range / extracted
    return "diamond"       # ◇ — primary composition (default)


def compute_functor_arity(tok: UdToken, tree: UdTree) -> int:
    """Estimate the number of remaining arguments for a functor node."""
    if tok.upos == "VERB":
        # Count core dependents (each is one argument slot).
        return len([d for d in tree.dependents_of(tok.id) if d.deprel in UD_CORE_DEPS])
    if tok.upos in ("ADP", "SCONJ"):
        return 1  # prepositions/complementizers take one argument
    return 0  # atomic


def ud_tree_to_mtlg(tree: UdTree) -> MtlgGraph:
    """Convert a UdTree to an MtlgGraph with modal mode and UCCA category assignments."""
    nodes = []
    for tok in tree.tokens:
        ucca_cat   = assign_ucca_category(tok)
        modal_mode = "diamond"  # nodes default to ◇; edges carry the actual mode
        arity      = compute_functor_arity(tok, tree)
        nodes.append(MtlgNode(
            token_id=tok.id, text=tok.text, lemma=tok.lemma,
            upos=tok.upos, modal_mode=modal_mode, category_id=ucca_cat, arity=arity,
        ))

    edges = []
    for tok in tree.tokens:
        if tok.head == 0:
            continue  # skip root
        head_tok = tree.token_by_id(tok.head)
        if head_tok is None:
            continue
        ucca_cat   = assign_ucca_category(tok)
        modal_mode = assign_modal_mode(tok, tok.deprel)
        edges.append(MtlgEdge(
            src_id=tok.head, dst_id=tok.id, deprel=tok.deprel,
            modal_mode=modal_mode, category_id=ucca_cat,
            arity=compute_functor_arity(head_tok, tree),
        ))

    return MtlgGraph(nodes=nodes, edges=edges, language=tree.language)


if __name__ == "__main__":
    import json
    from ud_parser import UdParser

    parser = UdParser("en")
    tree   = parser.parse("Alice runs quickly.")
    mtlg   = ud_tree_to_mtlg(tree)
    for edge in mtlg.edges:
        print(json.dumps({
            "src": edge.src_id, "dst": edge.dst_id,
            "deprel": edge.deprel, "mode": edge.modal_mode, "category_id": edge.category_id,
        }))
