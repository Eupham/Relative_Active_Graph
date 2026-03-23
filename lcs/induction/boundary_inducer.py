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


@dataclass
class ParseNode:
    """Intermediate lattice chunk during chart parsing."""
    label: str
    surface: str


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
        Hierarchical bottom-up lattice parse. 
        Iteratively applies rules matching RHS sequences to construct non-terminal LHS chunks.
        Newly grouped non-terminals act as inputs for higher-level CfgRule matches.
        """
        # Initialize lattice with raw code points (bottom layer)
        nodes: list[ParseNode] = [ParseNode(label=ch, surface=ch) for ch in text if ch.strip()]

        changed = True
        while changed:
            changed = False
            # Rules are inherently sorted by descending support from register_rule(), 
            # so highest-evidence hierarchical merges take topological precedence.
            for rule in self._rules:
                span = len(rule.rhs)
                if span == 0:
                    continue
                    
                i = 0
                while i + span <= len(nodes):
                    candidate_labels = [n.label for n in nodes[i:i + span]]
                    if candidate_labels == rule.rhs:
                        # Apply rule: collapse right-hand constituents into LHS non-terminal
                        merged_surface = "".join(n.surface for n in nodes[i:i + span])
                        new_node = ParseNode(label=rule.lhs, surface=merged_surface)
                        
                        # Rebuild lattice layer
                        nodes = nodes[:i] + [new_node] + nodes[i + span:]
                        changed = True
                        break # Reset to top-priority rule loop after mutation
                if changed:
                    break # Break out of rule iteration to retry from top support rule
                    
        return [
            ParseLeaf(id=idx + 1, text=n.surface, lemma=n.surface.lower())
            for idx, n in enumerate(nodes)
        ]
