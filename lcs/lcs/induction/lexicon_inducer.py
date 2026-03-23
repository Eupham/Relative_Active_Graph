"""
lexicon_inducer.py — MLE-based lexicon induction from UD-parsed C4 data.

Builds a per-lemma probability distribution over UCCA structural categories
from token frequency counts. Supports online updates from Rust-side attribution
deltas, closing the loop between inference-side learning and the induction side.

Output: en_lexicon.json (or {lang}_lexicon.json)
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _fnv_hash(s: str) -> int:
    h = 0x811c9dc5
    for b in s.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


class SuffixParadigmLattice:
    """
    Induces lemma equivalence classes from distributional context.
    Two surface forms are co-assigned a lemma_id when:
      cosine_sim(context_vector(form_a), context_vector(form_b)) >= threshold
    Context vectors are co-occurrence counts over a sliding window.
    """

    def __init__(self, window: int = 3, threshold: float = 0.85) -> None:
        self.window     = window
        self.threshold  = threshold
        self._contexts: dict[str, dict[str, int]] = {}   # form → neighbor counts
        self._lemma_ids: dict[str, int] = {}

    def observe(self, surface: str, neighbors: list[str]) -> None:
        c = self._contexts.setdefault(surface, {})
        for n in neighbors:
            c[n] = c.get(n, 0) + 1

    def commit(self) -> None:
        """Assign lemma_ids by merging forms with similar context vectors."""
        forms = list(self._contexts.keys())
        assigned: dict[str, int] = {}
        next_id = 1
        for i, form_a in enumerate(forms):
            if form_a in assigned:
                continue
            assigned[form_a] = next_id
            for form_b in forms[i + 1:]:
                if form_b not in assigned:
                    if self._cosine(form_a, form_b) >= self.threshold:
                        assigned[form_b] = next_id
            next_id += 1
        self._lemma_ids = assigned

    def lemma_id(self, surface: str) -> int:
        if surface in self._lemma_ids:
            return self._lemma_ids[surface]
        return _fnv_hash(surface) & 0xFFFFFFFF

    def _cosine(self, form_a: str, form_b: str) -> float:
        a = self._contexts.get(form_a, {})
        b = self._contexts.get(form_b, {})
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        dot   = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        norm_a = sum(v * v for v in a.values()) ** 0.5
        norm_b = sum(v * v for v in b.values()) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


@dataclass
class LexEntry:
    """One (lemma, ucca_cat) pair with frequency-derived probability."""
    lemma:       str
    ucca_cat:    str   # structural category label (e.g. "Process", "Participant")
    count:       float = 1.0
    probability: float = 0.0


class LexiconInducer:
    """
    Induces a per-lemma lexicon from UD-parsed corpus data via MLE counting.

    Usage:
        inducer = LexiconInducer()
        inducer.observe(lemma="run", ucca_cat="Process")
        inducer.normalise()
        inducer.save(Path("en_lexicon.json"))
    """

    def __init__(self) -> None:
        # lemma → list of LexEntry (one per observed ucca_cat)
        self._entries: dict[str, list[LexEntry]] = defaultdict(list)

    def observe(self, lemma: str, ucca_cat: str, count: float = 1.0) -> None:
        """Record an observation of `lemma` appearing in structural role `ucca_cat`."""
        entries = self._entries[lemma]
        existing = next((e for e in entries if e.ucca_cat == ucca_cat), None)
        if existing:
            existing.count += count
        else:
            entries.append(LexEntry(lemma=lemma, ucca_cat=ucca_cat, count=count))

    def normalise(self) -> None:
        """Convert raw counts to per-lemma probabilities."""
        for lemma, entries in self._entries.items():
            total = sum(e.count for e in entries) + 1e-12
            for e in entries:
                e.probability = e.count / total

    def online_update(self, attribution_deltas: dict[tuple[str, str], float]) -> None:
        """
        Apply attribution deltas from the Rust engine to lexicon entry weights.

        `attribution_deltas`: mapping of (lemma, ucca_cat_label) → signed delta.
        Positive delta: this (lemma, cat) combination was useful → increase weight.
        Negative delta: it was harmful → decrease weight.

        This replaces a full MLE re-count with an online Laplace-smoothed update.
        The delta is applied as a multiplicative perturbation on the count, keeping
        the relative ordering of categories while shifting mass toward useful ones.
        """
        for (lemma, cat_label), delta in attribution_deltas.items():
            if lemma not in self._entries:
                continue
            for entry in self._entries[lemma]:
                if entry.ucca_cat == cat_label:
                    # Shift count proportional to the existing count and the delta.
                    entry.count = max(0.01, entry.count + delta * entry.count)

        # Re-normalise totals after all deltas have been applied.
        self.normalise()

    def top_categories(self, lemma: str, n: int = 3) -> list[LexEntry]:
        """Return the top-N most probable categories for `lemma`."""
        entries = self._entries.get(lemma, [])
        return sorted(entries, key=lambda e: e.probability, reverse=True)[:n]

    def save(self, path: Path) -> None:
        data: dict[str, list[dict]] = {}
        for lemma, entries in self._entries.items():
            data[lemma] = [
                {
                    "ucca_cat":    e.ucca_cat,
                    "count":       e.count,
                    "probability": e.probability,
                }
                for e in entries
            ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Saved lexicon (%d lemmas) to %s", len(data), path)

    @classmethod
    def load(cls, path: Path) -> "LexiconInducer":
        inducer = cls()
        data = json.loads(path.read_text())
        for lemma, entries in data.items():
            for e in entries:
                inducer._entries[lemma].append(LexEntry(
                    lemma=lemma,
                    ucca_cat=e["ucca_cat"],
                    count=e["count"],
                    probability=e["probability"],
                ))
        return inducer


if __name__ == "__main__":
    import sys
    from pathlib import Path as P

    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    # Bootstrap from stub trees over a small C4 sample.
    try:
        sys.path.insert(0, str(P(__file__).parent))
        from mc4_stream import stream_mc4
    except ImportError:
        print("mc4_stream not available; cannot bootstrap lexicon.", file=sys.stderr)
        sys.exit(1)

    inducer = LexiconInducer()
    count = 0
    for item in stream_mc4(language=lang):
        if count >= samples:
            break
        text = item.get("text", "")
        for word in text.split():
            lemma = word.lower().strip(".,!?;:")
            if lemma:
                # Stub: all words observed as Participant (category discovery is Rust-side).
                inducer.observe(lemma=lemma, ucca_cat="Participant")
        count += 1

    inducer.normalise()
    out = P(f"{lang}_lexicon.json")
    inducer.save(out)
    print(f"Bootstrapped {len(inducer._entries)} lemmas → {out}")
