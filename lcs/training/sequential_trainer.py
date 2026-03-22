"""
sequential_trainer.py — End-to-end teacher-forcing trainer on C4 data.

Connects:
  - c4_sequence_extractor.py  → streams TokenSequence objects
  - rust_bridge.py            → submits them to the CSRRE engine
  - Reports CE-quality per epoch

Entry point:  python -m lcs.training.sequential_trainer [--lang en] [--trd 0] [--epochs 1]
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# ── Training config ───────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    language:       str   = "en"
    trd:            int   = 0
    epochs:         int   = 1
    max_sentences:  int   = 1_000
    log_interval:   int   = 100
    bootstrap_dir:  Optional[Path] = None
    binary:         Path  = Path("target/debug/csrre")


# ── Epoch statistics ──────────────────────────────────────────────────────────

@dataclass
class EpochStats:
    epoch:          int
    sentences:      int   = 0
    steps:          int   = 0
    quality_sum:    float = 0.0

    @property
    def mean_quality(self) -> float:
        return self.quality_sum / max(self.steps, 1)


# ── Main trainer ─────────────────────────────────────────────────────────────

class SequentialTrainer:
    """
    Teacher-forcing trainer that streams C4 sentences into the CSRRE engine.

    For each sentence:
      1. Extract TokenSequence (one step per token).
      2. Convert each step to WireNode + WireEdge for the engine.
      3. Call engine.execute_sequence (via CLI bridge).
      4. Accumulate CE-quality statistics.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._stats: list[EpochStats] = []

    def train(self) -> list[EpochStats]:
        from .rust_bridge import RustBridge
        from .c4_sequence_extractor import stream_c4_sequences, TokenSequence

        binary = self.config.binary
        if not binary.exists():
            # Try release build
            release = binary.parent.parent / "release" / binary.name
            if release.exists():
                binary = release
            else:
                logger.error("CSRRE binary not found at %s", binary)
                return []

        with RustBridge(binary=binary) as bridge:
            for epoch in range(self.config.epochs):
                stats = self._run_epoch(bridge, epoch)
                self._stats.append(stats)
                logger.info(
                    "Epoch %d: sentences=%d steps=%d mean_quality=%.4f",
                    epoch, stats.sentences, stats.steps, stats.mean_quality,
                )

        return self._stats

    def _run_epoch(
        self, bridge: "RustBridge", epoch: int
    ) -> EpochStats:
        from .c4_sequence_extractor import stream_c4_sequences

        stats = EpochStats(epoch=epoch)

        seq_iter = stream_c4_sequences(
            language=self.config.language,
            max_sentences=self.config.max_sentences,
        )

        for seq_count, seq in enumerate(seq_iter):
            stats.sentences += 1

            # Build node + edge pools for this sentence.
            # One node per token; one edge per consecutive pair.
            nodes = []
            edges = []
            for step in seq.steps:
                nodes.append(bridge.make_node(
                    node_id=step.node_id,
                    surface=step.text,
                    score=0.5,
                    cat=step.category_id,
                    arity=0,
                ))

            for i in range(len(seq.steps) - 1):
                src = seq.steps[i].node_id
                dst = seq.steps[i + 1].node_id
                eid = seq.steps[i].expected_edge_id
                edges.append(bridge.make_edge(edge_id=eid, src=src, dst=dst))

            # Submit sequence as individual queries (one per token step).
            # The engine accumulates CE-quality internally.
            for i, step in enumerate(seq.steps):
                step_nodes = nodes[max(0, i - 2) : i + 1]  # local context window
                result = bridge.query(
                    text=step.text,
                    situation_id=seq_count + 1,
                    trd=self.config.trd,
                    language=self.config.language,
                    nodes=step_nodes,
                    edges=edges[:i],
                )
                stats.steps      += 1
                stats.quality_sum += float(result.get("quality", 0.5))

            if stats.sentences % self.config.log_interval == 0:
                logger.info(
                    "  [epoch %d] %d sentences, mean_quality=%.4f",
                    epoch, stats.sentences, stats.mean_quality,
                )

        return stats


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="CSRRE sequential teacher-forcing trainer")
    parser.add_argument("--lang",     default="en",    help="mC4 language code")
    parser.add_argument("--trd",      type=int, default=0, help="TRD ID to train")
    parser.add_argument("--epochs",   type=int, default=1)
    parser.add_argument("--max-sents", type=int, default=1_000, dest="max_sentences")
    parser.add_argument("--binary",   type=Path, default=Path("target/debug/csrre"))
    parser.add_argument("--bootstrap-dir", type=Path, default=None, dest="bootstrap_dir")
    args = parser.parse_args(argv)

    config = TrainingConfig(
        language=args.lang,
        trd=args.trd,
        epochs=args.epochs,
        max_sentences=args.max_sentences,
        binary=args.binary,
        bootstrap_dir=args.bootstrap_dir,
    )

    trainer = SequentialTrainer(config)
    stats_list = trainer.train()

    if stats_list:
        final = stats_list[-1]
        print(f"Training complete: {final.sentences} sentences, "
              f"mean quality = {final.mean_quality:.4f}")
    else:
        print("Training produced no statistics (check binary path and data).")


if __name__ == "__main__":
    main()
