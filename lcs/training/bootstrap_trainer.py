"""
bootstrap_trainer.py — Phase 0 bootstrap: mC4 → UD → MTLG → cluster → export.

Runs the full bootstrap pipeline to produce:
  - categories.json   (cluster centroids + labels)
  - trd_profiles.json (modal type profiles per TRD)
  - edge_vocab.json   (edge ID → surface token mapping)

These artefacts are loaded by Rust's bootstrap_loader at training time.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# No hard-coded prototype labels: all category IDs are opaque u32 values
# assigned by the BIC-guided k-means bootstrap.  None carry linguistic names.


@dataclass
class CategoryRecord:
    id: int
    label: str
    centroid: list[float]
    count: int


@dataclass
class TrdProfileRecord:
    trd_id: int
    type_counts: dict[str, int]   # "mode_id,cat_id" → count
    total_tokens: int


@dataclass
class BootstrapArtefacts:
    categories:   list[CategoryRecord] = field(default_factory=list)
    trd_profiles: list[TrdProfileRecord] = field(default_factory=list)
    edge_vocab:   dict[str, str] = field(default_factory=dict)   # edge_id_str → surface


class BootstrapTrainer:
    """
    Orchestrates Phase 0: converts a small mC4 sample into bootstrap artefacts.

    Usage:
        trainer = BootstrapTrainer(output_dir=Path("bootstrap_out"))
        trainer.run(sentences=[...])
        trainer.save()
    """

    def __init__(self, output_dir: Path, n_clusters: int = 16):
        self.output_dir = output_dir
        self.n_clusters = n_clusters
        self.artefacts  = BootstrapArtefacts()
        self._next_edge_id = 1

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def run(self, sentences: list[str]) -> BootstrapArtefacts:
        """
        Full bootstrap from raw sentences.
        No UD parser. No pretrained model.
        """
        import sys, hashlib
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent / "induction"))
        import numpy as np
        from trd_bootstrap import (TrdBootstrapper, build_bigram_vocabulary,
                                    _sentence_to_bigram_vector)
        from c4_sequence_extractor import _sentence_to_sequence

        logger.info("Bootstrap: processing %d sentences", len(sentences))
        bootstrapper = TrdBootstrapper(n_clusters=self.n_clusters)
        trds = bootstrapper.bootstrap(sentences, language="bootstrap")

        # Categories from cluster centroids.
        if bootstrapper._model is not None and bootstrapper._model.centroids is not None:
            for i, centroid in enumerate(bootstrapper._model.centroids):
                self.artefacts.categories.append(CategoryRecord(
                    id=i + 1,
                    label=f"cluster_{i}",
                    centroid=centroid.tolist() if hasattr(centroid, "tolist") else list(centroid),
                    count=sum(1 for t in trds if t.trd_id == i),
                ))

        # TRD profiles from cluster bigram distributions.
        for trd in trds:
            type_counts: dict[str, int] = {}
            for bigram, freq in trd.modal_profile.items():
                cat_id = int(hashlib.sha256(bigram.encode()).hexdigest(), 16) % 65536
                key = f"0,{cat_id}"
                type_counts[key] = type_counts.get(key, 0) + int(freq * 1000)
            self.artefacts.trd_profiles.append(TrdProfileRecord(
                trd_id=trd.trd_id,
                type_counts=type_counts,
                total_tokens=trd.support,
            ))

        # Edge vocab from character sequences (first 500 sentences only).
        for sentence in sentences[:500]:
            seq = _sentence_to_sequence(sentence, trd_id=0)
            for step in seq.steps:
                eid = self._next_edge_id
                self.artefacts.edge_vocab[str(eid)] = step.text
                self._next_edge_id += 1

        return self.artefacts

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write artefact JSON files to output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        (self.output_dir / "categories.json").write_text(
            json.dumps([asdict(c) for c in self.artefacts.categories], indent=2)
        )
        (self.output_dir / "trd_profiles.json").write_text(
            json.dumps([asdict(p) for p in self.artefacts.trd_profiles], indent=2)
        )
        (self.output_dir / "edge_vocab.json").write_text(
            json.dumps(self.artefacts.edge_vocab, indent=2)
        )
        logger.info("Bootstrap artefacts saved to %s", self.output_dir)
