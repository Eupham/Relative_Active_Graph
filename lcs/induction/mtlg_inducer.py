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
    def __init__(self, language: str = "en", num_clusters: int = 10) -> None:
        self.language = language
        self.num_clusters = num_clusters
        self._entries: dict[str, list[LexEntry]] = defaultdict(list)
        # Contexts stores neighboring lemmas to induce categories structurally
        # Use explicit dict to avoid Pyre2 defaultdict-of-defaultdict inferrence errors
        self._contexts: dict[str, dict[str, float]] = {}

    def observe(self, lemma: str, _category_id: int, neighbor_lemma: str = "", count: float = 1.0) -> None:
        # Ignore predefined category_id! True unsupervised induction relies purely on structural context.
        if neighbor_lemma:
            if lemma not in self._contexts:
                self._contexts[lemma] = {}
            self._contexts[lemma][neighbor_lemma] = self._contexts[lemma].get(neighbor_lemma, 0.0) + count
        # Keep raw counts as baseline
        if not self._entries[lemma]:
            self._entries[lemma].append(LexEntry(lemma=lemma, ucca_cat="unknown", count=0))
        self._entries[lemma][0].count += count

    def _run_em_clustering(self) -> None:
        """
        Poisson Expectation-Maximization to cluster lemmas into latent syntactic categories.
        Language frequencies are discrete count data, so we model them via Poisson rates (lambda).
        """
        import random
        import math
        lemmas = list(self._contexts.keys())
        if not lemmas:
            return

        vocab = list({v for ctx in self._contexts.values() for v in ctx.keys()})
        # Initialize random counting rates (lambda)
        centroids = [{v: random.random() * 2.0 for v in vocab} for _ in range(self.num_clusters)]
        assignments = {l: random.randint(0, self.num_clusters - 1) for l in lemmas}

        for _iteration in range(5):  # EM loop
            # E-step: Assign lemmas via Poisson Log-Likelihood: k*log(lambda) - lambda
            # (k! term is constant wrt clusters, so omitted)
            for l in lemmas:
                best_c = 0
                best_score = -float('inf')
                for i, c in enumerate(centroids):
                    cluster_rate_sum = sum(c.values())
                    # Log PMF score
                    score = sum(self._contexts[l].get(v, 0) * math.log(c.get(v, 1e-12)) for v in self._contexts[l]) - cluster_rate_sum
                    if score > best_score:
                        best_score, best_c = score, i
                assignments[l] = best_c

            # M-step: Update lambda rates (MLE for Poisson rate is the average count across the assigned items)
            new_centroids = [{v: 1e-6 for v in vocab} for _ in range(self.num_clusters)]
            cluster_sizes = [0 for _ in range(self.num_clusters)]
            for l, c_idx in assignments.items():
                cluster_sizes[c_idx] += 1
                for v, count in self._contexts[l].items():
                    new_centroids[c_idx][v] += count
            
            for i, c in enumerate(new_centroids):
                size = max(cluster_sizes[i], 1)
                for k in c:
                    c[k] /= float(size)
            centroids = new_centroids

        # Update lexicon entries with unsupervised categories
        for l, c_idx in assignments.items():
            for entry in self._entries[l]:
                entry.ucca_cat = f"latent_cat_{c_idx}"
                entry.probability = 1.0

    def normalise(self) -> None:
        self._run_em_clustering()

    def best_type(self, lemma: str) -> Optional[LexEntry]:
        entries = self._entries.get(lemma, [])
        return entries[0] if entries else None

    def induce_from_stream(self, graphs: Iterator, max_trees: int = 10_000) -> "MtlgInducer":
        count = 0
        for g in graphs:
            if count >= max_trees:
                break
            # Extract structural co-occurrence pairs
            for edge in getattr(g, 'edges', []):
                src_node = next((n for n in g.nodes if getattr(n, 'token_id', -1) == edge.src_id), None)
                dst_node = next((n for n in g.nodes if getattr(n, 'token_id', -1) == edge.dst_id), None)
                if src_node and dst_node:
                    self.observe(getattr(src_node, 'lemma', ''), 0, getattr(dst_node, 'lemma', ''))
                    self.observe(getattr(dst_node, 'lemma', ''), 0, getattr(src_node, 'lemma', ''))
            # Fallback for unconnected nodes
            for node in getattr(g, 'nodes', []):
                self.observe(getattr(node, 'lemma', ''), 0)
            count += 1
        self.normalise()
        return self

    @property
    def entries(self) -> dict:
        return self._entries

    def save(self, path: Path) -> None:
        import math
        # Use log-link function to bridge discrete Poisson counts into continuous Gaussian causal weights
        data = {
            lemma: [{"ucca_cat": e.ucca_cat, "modal_mode": e.modal_mode,
                     "count": e.count, "probability": e.probability,
                     "causal_weight": math.log1p(e.count)} for e in entries]
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
