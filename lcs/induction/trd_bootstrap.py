"""TRD bootstrapping: crystallize co-occurring UCCA-category type profiles into TRD entries.

A TRD (Transient Relative Domain) is a cluster of situation types, infon patterns,
and MTLG modal type profiles that co-activate together.

Algorithm:
  1. Process dissolved TRs from the LCS pipeline (per language).
  2. Collect (modal_mode, ucca_cat) co-occurrence vectors per sentence/context.
  3. Cluster co-occurrence vectors (k-means over the modal profile space).
  4. Each cluster → one TRD entry: {type_profile, modal_operator_inventory}.
  5. Validate: > 70% of held-out situations map to a known TRD.

TRD vocabulary target: < 1000 entries per language family (to prevent fragmentation).
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ud_to_mtlg import MtlgGraph

logger = logging.getLogger(__name__)

MAX_TRD_VOCAB = 1000
N_CLUSTERS_DEFAULT = 64  # starting point; tuned via held-out coverage


@dataclass
class TrdEntry:
    trd_id:      int
    label:       str
    modal_profile: dict[str, float]   # (mode, cat) key → frequency
    infon_patterns: list[str]
    support:     int = 0

    def to_dict(self) -> dict:
        return {
            "trd_id": self.trd_id,
            "label":  self.label,
            "modal_profile": self.modal_profile,
            "infon_patterns": self.infon_patterns,
            "support": self.support,
        }


def graph_to_modal_vector(graph: MtlgGraph, vocab: list[str]) -> np.ndarray:
    """Convert a graph's modal type distribution to a fixed-size vector."""
    counts: Counter = Counter()
    for edge in graph.edges:
        key = f"{edge.modal_mode}_{edge.ucca_cat}"
        counts[key] += 1
    total = sum(counts.values()) or 1
    return np.array([counts.get(v, 0) / total for v in vocab], dtype=np.float32)


def build_vocabulary(graphs: list[MtlgGraph]) -> list[str]:
    """Build the modal-profile vocabulary from observed (mode, cat) pairs."""
    keys: set = set()
    for g in graphs:
        for e in g.edges:
            keys.add(f"{e.modal_mode}_{e.ucca_cat}")
    return sorted(keys)


class KMeansTrd:
    """Simple k-means clustering for TRD crystallization."""

    def __init__(self, n_clusters: int, max_iter: int = 20, seed: int = 42):
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.rng        = np.random.default_rng(seed)
        self.centroids: np.ndarray | None = None

    def fit(self, vectors: np.ndarray) -> np.ndarray:
        """Returns cluster assignments for each vector."""
        n = len(vectors)
        k = min(self.n_clusters, n)
        # Initialize centroids with k-means++ heuristic.
        idx = [self.rng.integers(0, n)]
        for _ in range(k - 1):
            dists = np.array([min(np.linalg.norm(v - vectors[i]) ** 2 for i in idx) for v in vectors])
            probs = dists / dists.sum()
            idx.append(self.rng.choice(n, p=probs))
        self.centroids = vectors[idx]

        assignments = np.zeros(n, dtype=int)
        for _ in range(self.max_iter):
            # Assign each vector to nearest centroid.
            dists = np.linalg.norm(vectors[:, None] - self.centroids[None, :], axis=2)
            new_assignments = dists.argmin(axis=1)
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments
            # Update centroids.
            for c in range(k):
                members = vectors[assignments == c]
                if len(members) > 0:
                    self.centroids[c] = members.mean(axis=0)
        return assignments

    def predict(self, vector: np.ndarray) -> int:
        if self.centroids is None:
            raise RuntimeError("Fit the model first.")
        dists = np.linalg.norm(self.centroids - vector, axis=1)
        return int(dists.argmin())


class TrdBootstrapper:
    def __init__(self, n_clusters: int = N_CLUSTERS_DEFAULT):
        self.n_clusters = n_clusters
        self.vocab:    list[str] = []
        self.trds:     list[TrdEntry] = []
        self._model:   KMeansTrd | None = None

    def bootstrap(self, graphs: list[MtlgGraph], language: str) -> list[TrdEntry]:
        """Crystallize TRDs from a batch of MTLG graphs."""
        if not graphs:
            return []
        self.vocab = build_vocabulary(graphs)
        vectors    = np.stack([graph_to_modal_vector(g, self.vocab) for g in graphs])
        n_clusters = min(self.n_clusters, len(graphs), MAX_TRD_VOCAB)
        self._model = KMeansTrd(n_clusters)
        assignments = self._model.fit(vectors)
        self.trds = self._build_trd_entries(graphs, assignments, n_clusters, language)
        logger.info("Bootstrapped %d TRDs for %s from %d graphs", len(self.trds), language, len(graphs))
        return self.trds

    def _build_trd_entries(
        self, graphs: list[MtlgGraph], assignments: np.ndarray,
        n_clusters: int, language: str,
    ) -> list[TrdEntry]:
        cluster_graphs: dict[int, list[MtlgGraph]] = defaultdict(list)
        for i, g in enumerate(graphs):
            cluster_graphs[int(assignments[i])].append(g)

        trds = []
        for cluster_id, cg in cluster_graphs.items():
            # Build modal profile from cluster centroid.
            centroid = self._model.centroids[cluster_id]
            profile  = {v: float(centroid[i]) for i, v in enumerate(self.vocab) if centroid[i] > 0.01}
            # Characteristic infon patterns: most common lemmas in this cluster.
            lemma_counter: Counter = Counter()
            for g in cg:
                for node in g.nodes:
                    if node.ucca_cat in ("Process", "Participant"):
                        lemma_counter[node.lemma] += 1
            top_lemmas = [l for l, _ in lemma_counter.most_common(5)]
            trds.append(TrdEntry(
                trd_id=cluster_id,
                label=f"{language}_trd_{cluster_id}",
                modal_profile=profile,
                infon_patterns=top_lemmas,
                support=len(cg),
            ))
        return trds

    def coverage(self, held_out: list[MtlgGraph]) -> float:
        """Fraction of held-out graphs that map to a known TRD."""
        if not held_out or self._model is None:
            return 0.0
        matched = 0
        for g in held_out:
            vec = graph_to_modal_vector(g, self.vocab)
            trd_id = self._model.predict(vec)
            if any(t.trd_id == trd_id for t in self.trds):
                matched += 1
        return matched / len(held_out)

    def save(self, path: Path):
        data = {
            "vocab": self.vocab,
            "trds":  [t.to_dict() for t in self.trds],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Saved %d TRDs to %s", len(self.trds), path)


if __name__ == "__main__":
    import sys
    from ud_to_mtlg import ud_tree_to_mtlg
    from ud_parser import UdParser
    from mc4_stream import stream_mc4
    from morphological_fst import preprocess_for_type_assignment

    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    parser = UdParser(lang)
    graphs = []
    for item in stream_mc4(lang, max_samples=200):
        try:
            tokens   = preprocess_for_type_assignment(item["text"], lang)
            sentence = " ".join(t.split("[")[0] for t in tokens)
            tree     = parser.parse(sentence)
            graphs.append(ud_tree_to_mtlg(tree))
        except Exception:
            pass

    train, held_out = graphs[:150], graphs[150:]
    bootstrapper = TrdBootstrapper(n_clusters=16)
    trds = bootstrapper.bootstrap(train, lang)
    cov  = bootstrapper.coverage(held_out)
    print(f"TRDs: {len(trds)}, coverage on held-out: {cov:.2%}")
    bootstrapper.save(Path(f"{lang}_trds.json"))
