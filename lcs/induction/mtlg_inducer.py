"""Probabilistic MTLG grammar induction.

Extends Kwiatkowski et al. (2010) higher-order unification to multimodal types.
Bisk & Hockenmaier HDP-CCG extended to modal categories (◇, □, ◊).

Pipeline:
  UD tree → MTLG graph → extract (word, modal_type) pairs → update lexicon counts → normalize.

Output: per-language modal lexicon stored as JSON.
  {"lemma": {"modal_mode": "diamond", "ucca_cat": "Process", "arity": 2, "count": 145}, ...}
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ud_to_mtlg import MtlgGraph, MtlgNode, ud_tree_to_mtlg
from morphological_fst import preprocess_for_type_assignment

logger = logging.getLogger(__name__)


@dataclass
class LexTypeEntry:
    modal_mode: str
    ucca_cat:   str
    arity:      int
    count:      int = 0
    log_prob:   float = 0.0


@dataclass
class PerLanguageLexicon:
    language: str
    entries:  dict[str, list[LexTypeEntry]] = field(default_factory=lambda: defaultdict(list))

    def update(self, lemma: str, mode: str, cat: str, arity: int, weight: float = 1.0):
        for entry in self.entries[lemma]:
            if entry.modal_mode == mode and entry.ucca_cat == cat and entry.arity == arity:
                entry.count += weight
                return
        self.entries[lemma].append(LexTypeEntry(mode, cat, arity, count=weight))

    def normalize(self):
        """Compute log-probabilities from counts (MLE)."""
        for lemma, type_entries in self.entries.items():
            total = sum(e.count for e in type_entries)
            if total == 0:
                continue
            for e in type_entries:
                e.log_prob = math.log(e.count / total)

    def best_type(self, lemma: str) -> LexTypeEntry | None:
        entries = self.entries.get(lemma)
        if not entries:
            return None
        return max(entries, key=lambda e: e.count)

    def to_dict(self) -> dict:
        return {
            lemma: [
                {"modal_mode": e.modal_mode, "ucca_cat": e.ucca_cat,
                 "arity": e.arity, "count": e.count, "log_prob": e.log_prob}
                for e in entries
            ]
            for lemma, entries in self.entries.items()
        }

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))
        logger.info("Saved lexicon for %s to %s (%d lemmas)", self.language, path, len(self.entries))

    @classmethod
    def load(cls, language: str, path: Path) -> "PerLanguageLexicon":
        data = json.loads(path.read_text())
        lex = cls(language=language)
        for lemma, entries in data.items():
            for e in entries:
                lex.entries[lemma].append(LexTypeEntry(
                    e["modal_mode"], e["ucca_cat"], e["arity"],
                    count=e["count"], log_prob=e["log_prob"],
                ))
        return lex


class MtlgInducer:
    """Induces a probabilistic MTLG lexicon from MTLG graphs derived from UD trees."""

    def __init__(self, language: str):
        self.language = language
        self.lexicon  = PerLanguageLexicon(language=language)
        self._trees_processed = 0

    def observe_graph(self, graph: MtlgGraph):
        """Update lexicon counts from one MTLG graph."""
        for node in graph.nodes:
            # Weight by: arity > 0 gives more signal (functors)
            weight = 1.5 if node.arity > 0 else 1.0
            self.lexicon.update(node.lemma, node.modal_mode, node.ucca_cat, node.arity, weight)
        self._trees_processed += 1

    def induce_from_stream(
        self,
        trees: Iterator[MtlgGraph],
        max_trees: int = 10_000,
    ) -> PerLanguageLexicon:
        """Induce lexicon from a stream of MTLG graphs."""
        for i, graph in enumerate(trees):
            if i >= max_trees:
                break
            self.observe_graph(graph)
            if i % 1_000 == 0:
                logger.info("Processed %d trees for %s", i, self.language)
        self.lexicon.normalize()
        logger.info("Induced %d lemmas for %s from %d trees",
                    len(self.lexicon.entries), self.language, self._trees_processed)
        return self.lexicon

    def parse_accuracy(self, test_graphs: list[MtlgGraph]) -> float:
        """Estimate parse accuracy: fraction of nodes where best-predicted type matches gold."""
        if not test_graphs:
            return 0.0
        correct = total = 0
        for graph in test_graphs:
            for node in graph.nodes:
                total += 1
                best = self.lexicon.best_type(node.lemma)
                if best and best.modal_mode == node.modal_mode and best.ucca_cat == node.ucca_cat:
                    correct += 1
        return correct / total if total > 0 else 0.0


def run_induction_pipeline(
    language:    str,
    max_samples: int = 5_000,
    output_dir:  str = "lexicons",
) -> PerLanguageLexicon:
    """End-to-end induction: mC4 stream → UD parse → MTLG graph → lexicon."""
    from mc4_stream import stream_mc4, requires_fst
    from ud_parser import UdParser

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parser  = UdParser(language)
    inducer = MtlgInducer(language)

    def graph_stream():
        for item in stream_mc4(language, max_samples=max_samples):
            tokens = preprocess_for_type_assignment(item["text"], language)
            sentence = " ".join(t.split("[")[0] for t in tokens)  # strip FST tags for parser
            try:
                tree  = parser.parse(sentence)
                graph = ud_tree_to_mtlg(tree)
                yield graph
            except Exception as exc:
                logger.debug("Parse error: %s", exc)

    lex = inducer.induce_from_stream(graph_stream(), max_trees=max_samples)
    lex.save(out_path / f"{language}_lexicon.json")
    return lex


if __name__ == "__main__":
    import sys
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    lex  = run_induction_pipeline(lang, max_samples=100)
    print(f"Induced {len(lex.entries)} lemmas for {lang}")
    # Show top 5 by count
    top = sorted(lex.entries.items(), key=lambda kv: sum(e.count for e in kv[1]), reverse=True)[:5]
    for lemma, entries in top:
        best = max(entries, key=lambda e: e.count)
        print(f"  {lemma}: mode={best.modal_mode} cat={best.ucca_cat} arity={best.arity} count={best.count:.0f}")
