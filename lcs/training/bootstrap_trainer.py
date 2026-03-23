"""
bootstrap_trainer.py — Phase 0 bootstrap: mC4 → MTLG → community induction → export.

Runs the full bootstrap pipeline to produce:
  - categories.json   (community descriptors + sizes)
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


def _fnv_hash(s: str) -> int:
    h = 0x811c9dc5
    for b in s.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


# No hard-coded prototype labels: all category IDs are opaque u32 values
# assigned by the SBM Leiden MDL bootstrap.  None carry linguistic names.


@dataclass
class CategoryRecord:
    id:         int
    descriptor: list[str]   # top-5 surface forms (cluster descriptor)
    size:       int


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

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.artefacts  = BootstrapArtefacts()
        self._next_edge_id = 1

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def run(self, sentences: list[str]) -> BootstrapArtefacts:
        """
        Full bootstrap from raw sentences.
        No UD parser. No pretrained model. No k-means.
        Uses SBM Leiden MDL community induction.
        """
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent / "induction"))
        from community_inducer import CommunityInducer
        from c4_sequence_extractor import _sentence_to_sequence

        logger.info("Bootstrap: processing %d sentences", len(sentences))

        # Build co-occurrence data from sentences.
        inducer = CommunityInducer(window=3)
        for sentence in sentences:
            tokens = sentence.split()
            inducer.observe_sentence(tokens)

        # Fit communities via SBM Leiden MDL.
        communities = inducer.fit()

        # Categories from community descriptors.
        for community in communities:
            self.artefacts.categories.append(CategoryRecord(
                id=community.id,
                descriptor=community.top5,
                size=community.size,
            ))

        # TRD profiles from community bigram distributions.
        for community in communities:
            type_counts: dict[str, int] = {}
            for surface in community.members:
                # Use FNV hash for stable, deterministic bigram IDs (§20.3).
                cat_id = _fnv_hash(surface) & 0xFFFF
                key = f"0,{cat_id}"
                type_counts[key] = type_counts.get(key, 0) + 1
            self.artefacts.trd_profiles.append(TrdProfileRecord(
                trd_id=community.id,
                type_counts=type_counts,
                total_tokens=community.size,
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
