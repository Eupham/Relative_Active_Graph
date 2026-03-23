"""
boundary_inducer.py — Grammar-driven boundary induction.
No prior rules. Token boundaries are structural artifacts of parse sub-trees.
The MetaGrammar Engine registers rules via register_rule().
Before any rules exist, every code point is its own ParseLeaf.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CfgRule:
    """A production rule discovered by the MetaGrammar Engine."""
    id:          int
    lhs:         str          # non-terminal label
    rhs:         list[str]    # sequence of non-terminal or terminal labels
    support:     float = 1.0  # evidence count


@dataclass
class ParseLeaf:
    """A terminal parse node: the output unit of boundary induction."""
    id:     int
    text:   str
    lemma:  str   # lowercased form; suffix-paradigm lattice updates this (§0.3)


class BoundaryInducer:
    """
    Inductive boundary inducer. Zero prior knowledge at initialization.
    Rules grow from evidence supplied by the MetaGrammar Engine.

    Interface contract:
        inducer = BoundaryInducer()
        leaves  = inducer.parse(text)   # returns list[ParseLeaf]
        inducer.register_rule(rule)     # called by MetaGrammar Engine
    """

    def __init__(self) -> None:
        self._rules: list[CfgRule] = []

    def register_rule(self, rule: CfgRule) -> None:
        """Register a new structural rule. Called by MetaGrammar Engine."""
        existing = next((r for r in self._rules
                         if r.lhs == rule.lhs and r.rhs == rule.rhs), None)
        if existing:
            existing.support += 1.0
        else:
            self._rules.append(rule)
        # Sort by descending support so highest-evidence rules apply first.
        self._rules.sort(key=lambda r: r.support, reverse=True)

    def parse(self, text: str) -> list[ParseLeaf]:
        """
        Return ParseLeaf objects for the input text.
        Without rules: one leaf per code point.
        With rules: chart parse groups code points per discovered structure.
        """
        if not self._rules:
            return self._codepoint_leaves(text)
        return self._chart_parse(text)

    def _codepoint_leaves(self, text: str) -> list[ParseLeaf]:
        return [
            ParseLeaf(id=i + 1, text=ch, lemma=ch.lower())
            for i, ch in enumerate(text)
            if ch.strip()  # skip bare whitespace at top level
        ] or [ParseLeaf(id=1, text=text, lemma=text.lower())]

    def _chart_parse(self, text: str) -> list[ParseLeaf]:
        """
        Greedy left-to-right chart parse using self._rules.
        Rules are tried in descending support order; longest matching RHS wins.
        Falls back to single code point when no rule applies.
        """
        chars = list(text)
        leaves: list[ParseLeaf] = []
        i = 0
        leaf_id = 1
        while i < len(chars):
            matched = False
            for rule in self._rules:
                span = len(rule.rhs)
                if i + span <= len(chars):
                    candidate = chars[i:i + span]
                    if candidate == rule.rhs:
                        surface = "".join(candidate)
                        leaves.append(ParseLeaf(id=leaf_id, text=surface,
                                                lemma=surface.lower()))
                        leaf_id += 1
                        i += span
                        matched = True
                        break
            if not matched:
                ch = chars[i]
                if ch.strip():
                    leaves.append(ParseLeaf(id=leaf_id, text=ch, lemma=ch.lower()))
                    leaf_id += 1
                i += 1
        return leaves
