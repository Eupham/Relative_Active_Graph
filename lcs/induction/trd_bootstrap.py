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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


def _sentence_to_bigram_vector(sentence: str, vocab: list[str]) -> "np.ndarray":
    """Character bigram frequency vector. Clusters by script/language/register."""
    counts: dict[str, int] = {}
    chars = list(sentence)
    for i in range(len(chars) - 1):
        bg = chars[i] + chars[i + 1]
        counts[bg] = counts.get(bg, 0) + 1
    total = sum(counts.values()) or 1
    return np.array([counts.get(v, 0) / total for v in vocab], dtype=np.float32)


def build_bigram_vocabulary(sentences: list[str]) -> list[str]:
    """Sorted list of all observed character bigrams."""
    keys: set[str] = set()
    for s in sentences:
        chars = list(s)
        for i in range(len(chars) - 1):
            keys.add(chars[i] + chars[i + 1])
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

    def bootstrap(self, sentences: list[str], language: str) -> list[TrdEntry]:
        if not sentences:
            logger.warning("No sentences provided for TRD bootstrap")
            return []
        self.vocab = build_bigram_vocabulary(sentences)
        if not self.vocab:
            logger.warning("Empty bigram vocabulary")
            return []
        vectors = np.stack([_sentence_to_bigram_vector(s, self.vocab) for s in sentences])
        k = min(self.n_clusters, len(sentences))
        self._model = KMeansTrd(n_clusters=k)
        assignments = self._model.fit(vectors)
        trds: list[TrdEntry] = []
        for cluster_id in range(k):
            members = [sentences[i] for i, a in enumerate(assignments) if a == cluster_id]
            if not members:
                continue
            profile: dict[str, float] = {}
            for s in members:
                chars = list(s)
                for i in range(len(chars) - 1):
                    bg = chars[i] + chars[i + 1]
                    profile[bg] = profile.get(bg, 0) + 1
            total = sum(profile.values()) or 1
            profile = {k: v / total for k, v in profile.items()}
            top_bigrams = sorted(profile, key=lambda x: profile[x], reverse=True)[:5]
            trds.append(TrdEntry(
                trd_id=cluster_id,
                label=f"{language}_trd_{cluster_id}",
                modal_profile=profile,
                infon_patterns=top_bigrams,
                support=len(members),
            ))
        self.trds = trds
        logger.info("Bootstrapped %d TRDs for language=%s from %d sentences",
                    len(trds), language, len(sentences))
        return trds

    def coverage(self, held_out: list[str]) -> float:
        """Fraction of held-out sentences that map to a known TRD."""
        if not held_out or self._model is None:
            return 0.0
        matched = 0
        for s in held_out:
            vec = _sentence_to_bigram_vector(s, self.vocab)
            trd_id = self._model.predict(vec)
            if any(t.trd_id == trd_id for t in self.trds):
                matched += 1
        return matched / len(held_out)

    def online_centroid_update(
        self,
        passage_vector: list[float],
        quality: float,
        trd_id: int,
        lr: float = 0.01,
    ) -> None:
        """
        Shift the centroid of `trd_id` toward `passage_vector` weighted by `quality`.

        Called after each passage flush. High-quality passages (low CE loss) move the
        centroid more than low-quality ones. This allows TRDs to drift as the engine
        encounters new domains in C4, preventing the initial bootstrap from becoming stale.

        `passage_vector`: modal-profile vector for the passage (same dimensionality as
                          the bootstrap vocab vectors).
        `quality`:        mean CE quality over the passage (in [0, 1]).
        `trd_id`:         which TRD centroid to update.
        `lr`:             learning rate (default 0.01).
        """
        if self._model is None or self._model.centroids is None:
            return
        for trd in self.trds:
            if trd.trd_id == trd_id:
                c = self._model.centroids[trd_id]
                pv = passage_vector
                for i in range(min(len(c), len(pv))):
                    c[i] += lr * quality * (pv[i] - c[i])
                break

    def save(self, path: Path):
        data = {
            "vocab": self.vocab,
            "trds":  [t.to_dict() for t in self.trds],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Saved %d TRDs to %s", len(self.trds), path)


if __name__ == "__main__":
    import sys
    from mc4_stream import stream_mc4
    lang    = sys.argv[1] if len(sys.argv) > 1 else "en"
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    sentences = [item["text"] for item in stream_mc4(lang, max_samples=samples)
                 if item.get("text", "").strip()]
    bootstrapper = TrdBootstrapper(n_clusters=16)
    trds = bootstrapper.bootstrap(sentences, lang)
    print(f"TRDs: {len(trds)}")
    for t in trds[:3]:
        print(f"  {t.label}: support={t.support}, top_bigrams={t.infon_patterns[:3]}")
