"""
bootstrap_trainer.py — Phase 0 bootstrap: mC4 → UD → MTLG → cluster → export.

Runs the full bootstrap pipeline to produce:
  - categories.json   (cluster centroids + labels)
  - trd_profiles.json (modal type profiles per TRD)
  - edge_vocab.json   (edge ID → surface token mapping)

These artefacts are loaded by Rust's bootstrap_loader at training time.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ── Structural prototype labels (mirror Rust constants) ──────────────────────

PROTOTYPE_LABELS = {
    1: "Process",
    2: "Connector",
    3: "Ground",
    4: "Adverbial",
    5: "State",
    6: "Participant",
}


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
        Full bootstrap from a list of raw sentences.

        Steps:
        1. Parse sentences with UD parser (stanza / spacy-dep).
        2. Convert UD trees → MtlgGraph (structural evidence only).
        3. Cluster deferred tokens via CategoryInducer.
        4. Build TRD profiles from co-activation patterns.
        5. Build edge vocab from surface tokens.
        """
        from .c4_sequence_extractor import parse_ud_trees

        logger.info("Bootstrap: parsing %d sentences", len(sentences))
        ud_trees = parse_ud_trees(sentences)

        # Import the Rust-side converter (via Python cffi or re-implementation).
        # Here we use the Python re-implementation from lcs/induction.
        try:
            from lcs.induction.ud_to_mtlg import UdToMtlgConverter
            converter = UdToMtlgConverter(n_clusters=self.n_clusters)
            mtlg_graphs = [converter.convert(tree) for tree in ud_trees]
            converter.resolve_deferred(mtlg_graphs)
        except ImportError:
            logger.warning("lcs.induction.ud_to_mtlg not available; using stub graphs")
            mtlg_graphs = []

        self._build_categories(converter if mtlg_graphs else None)
        self._build_trd_profiles(mtlg_graphs)
        self._build_edge_vocab(mtlg_graphs)
        return self.artefacts

    def _build_categories(self, converter: Any) -> None:
        """Populate category records from cluster centroids."""
        # Structural prototypes (fixed IDs 1–6).
        for pid, label in PROTOTYPE_LABELS.items():
            self.artefacts.categories.append(CategoryRecord(
                id=pid, label=label, centroid=[], count=0
            ))

        # Discovered clusters from CategoryInducer.
        if converter is not None:
            try:
                inducer = converter.inducer
                for c_idx, centroid in enumerate(inducer.centroids_):
                    label = f"cluster_{c_idx + 7}"
                    count = int(inducer.counts_.get(c_idx, 0))
                    self.artefacts.categories.append(CategoryRecord(
                        id=c_idx + 7,
                        label=label,
                        centroid=centroid.tolist(),
                        count=count,
                    ))
            except AttributeError:
                pass

    def _build_trd_profiles(self, graphs: list) -> None:
        """
        Build TRD profiles: co-activation of (mode, category) pairs.

        Simple heuristic: TRD ID = hash(dominant_category) % 8.
        """
        from collections import defaultdict
        profiles: dict[int, TrdProfileRecord] = {}

        for graph in graphs:
            for node in getattr(graph, "nodes", []):
                cat_id  = getattr(node, "category_id", 0)
                mode_id = 0  # Diamond default
                trd_id  = cat_id % 8
                if trd_id not in profiles:
                    profiles[trd_id] = TrdProfileRecord(
                        trd_id=trd_id, type_counts={}, total_tokens=0
                    )
                key = f"{mode_id},{cat_id}"
                profiles[trd_id].type_counts[key] = (
                    profiles[trd_id].type_counts.get(key, 0) + 1
                )
                profiles[trd_id].total_tokens += 1

        self.artefacts.trd_profiles = list(profiles.values())

    def _build_edge_vocab(self, graphs: list) -> None:
        """Map edge IDs to surface tokens."""
        for graph in graphs:
            for edge in getattr(graph, "edges", []):
                eid  = self._next_edge_id
                surf = getattr(edge, "surface", "") or ""
                self.artefacts.edge_vocab[str(eid)] = surf
                self._next_edge_id += 1

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
