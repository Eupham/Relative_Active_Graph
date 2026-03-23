"""MLE-based lexicon induction from tokenised text. No UD field references."""
from __future__ import annotations
import json, logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class LexEntry:
    lemma:       str
    ucca_cat:    str
    modal_mode:  str   = "diamond"
    count:       float = 1.0
    probability: float = 0.0


class MtlgInducer:
    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._entries: dict[str, list[LexEntry]] = defaultdict(list)

    def observe(self, lemma: str, category_id: int, count: float = 1.0) -> None:
        cat_label = f"cluster_{category_id}"
        entries   = self._entries[lemma]
        existing  = next((e for e in entries if e.ucca_cat == cat_label), None)
        if existing:
            existing.count += count
        else:
            entries.append(LexEntry(lemma=lemma, ucca_cat=cat_label, count=count))

    def normalise(self) -> None:
        for entries in self._entries.values():
            total = sum(e.count for e in entries) + 1e-12
            for e in entries:
                e.probability = e.count / total

    def best_type(self, lemma: str) -> Optional[LexEntry]:
        entries = self._entries.get(lemma, [])
        return max(entries, key=lambda e: e.probability if e.probability > 0 else e.count) if entries else None

    def induce_from_stream(self, graphs: Iterator, max_trees: int = 10_000) -> "MtlgInducer":
        count = 0
        for g in graphs:
            if count >= max_trees:
                break
            for node in g.nodes:
                self.observe(node.lemma, node.category_id)
            count += 1
        self.normalise()
        return self

    @property
    def entries(self) -> dict:
        return self._entries

    def save(self, path: Path) -> None:
        data = {
            lemma: [{"ucca_cat": e.ucca_cat, "modal_mode": e.modal_mode,
                     "count": e.count, "probability": e.probability} for e in entries]
            for lemma, entries in self._entries.items()
        }
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        for lemma, raw in data.items():
            for r in raw:
                self._entries[lemma].append(LexEntry(
                    lemma=lemma, ucca_cat=r["ucca_cat"],
                    modal_mode=r.get("modal_mode", "diamond"),
                    count=r["count"], probability=r["probability"],
                ))
