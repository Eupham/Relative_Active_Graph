"""
sequential_trainer.py — End-to-end teacher-forcing trainer on C4 data.

Connects:
  - c4_sequence_extractor.py  → streams TokenSequence objects, grouped into passages
  - rust_bridge.py            → submits passages to the CSRRE engine
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

# Target passage size in characters. Sentences are accumulated until this
# threshold is exceeded, then the passage is flushed as a single training unit.
TARGET_PASSAGE_CHARS = 2000


# ── Training config ───────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    language:              str   = "en"
    trd:                   int   = 0
    epochs:                int   = 1
    max_sentences:         int   = 1_000
    log_interval:          int   = 100
    passage_chars:         int   = TARGET_PASSAGE_CHARS
    bootstrap_dir:         Optional[Path] = None
    binary:                Path  = Path("target/debug/csrre")
    online_lexicon_update: bool  = False
    online_trd_update:     bool  = False


# ── Epoch statistics ──────────────────────────────────────────────────────────

@dataclass
class EpochStats:
    epoch:          int
    sentences:      int   = 0
    steps:          int   = 0
    quality_sum:    float = 0.0
    passages:       int   = 0

    @property
    def mean_quality(self) -> float:
        return self.quality_sum / max(self.steps, 1)


# ── Main trainer ─────────────────────────────────────────────────────────────

class SequentialTrainer:
    """
    Passage-level teacher-forcing trainer that streams C4 into the CSRRE engine.

    For each passage (~2000 characters of consecutive sentences):
      1. Extract TokenSequence for each sentence (one step per token, no hard labels).
      2. Convert each sentence's steps to WireNode + WireEdge for the engine.
      3. Call engine.execute_passage (via CLI bridge) — processes the full passage
         as a single training unit with cross-sentence attribution.
      4. Accumulate CE-quality statistics.
      5. Optionally update lexicon and TRD centroids online.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._stats: list[EpochStats] = []

    def train(self) -> list[EpochStats]:
        from .rust_bridge import RustBridge

        binary = self.config.binary
        if not binary.exists():
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
                    "Epoch %d: passages=%d sentences=%d steps=%d mean_quality=%.4f",
                    epoch, stats.passages, stats.sentences, stats.steps, stats.mean_quality,
                )

        return self._stats

    def _run_epoch(self, bridge: "RustBridge", epoch: int) -> EpochStats:
        from .c4_sequence_extractor import stream_c4_sequences

        stats = EpochStats(epoch=epoch)

        # Passage accumulator: list of TokenSequence objects
        passage_buffer: list = []
        passage_chars = 0

        def flush_passage() -> None:
            nonlocal passage_chars
            if not passage_buffer:
                return

            # Convert each sentence's TokenSequence to wire format.
            wire_sentences = []
            for seq in passage_buffer:
                nodes = [bridge.make_node(
                    node_id=s.node_id,
                    surface=s.text,
                    score=0.5,
                    deprel_hash=s.deprel_hash,
                    upos_hash=s.upos_hash,
                ) for s in seq.steps]

                edges = [bridge.make_edge(
                    edge_id=i + 1,
                    src=seq.steps[i].node_id,
                    dst=seq.steps[i + 1].node_id,
                ) for i in range(len(seq.steps) - 1)]

                wire_sentences.append({
                    "nodes": nodes,
                    "edges": edges,
                    "steps": [s.expected_node_id for s in seq.steps],
                })

            result = bridge.execute_passage(
                trd=self.config.trd,
                language=self.config.language,
                sentences=wire_sentences,
            )

            stats.passages      += 1
            stats.sentences     += len(passage_buffer)
            stats.steps         += int(result.get("steps_processed", 0))
            stats.quality_sum   += float(result.get("quality_sum", 0.0))

            passage_buffer.clear()
            passage_chars = 0  # reset via nonlocal capture

        seq_iter = stream_c4_sequences(
            language=self.config.language,
            max_sentences=self.config.max_sentences,
        )

        for seq in seq_iter:
            passage_buffer.append(seq)
            passage_chars += len(seq.sentence)

            if passage_chars >= self.config.passage_chars:
                flush_passage()

            if stats.passages > 0 and stats.passages % self.config.log_interval == 0:
                logger.info(
                    "  [epoch %d] %d passages, mean_quality=%.4f",
                    epoch, stats.passages, stats.mean_quality,
                )

        # Flush any remaining sentences as a final partial passage.
        flush_passage()

        return stats


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="CSRRE sequential teacher-forcing trainer")
    parser.add_argument("--lang",     default="en",    help="mC4 language code")
    parser.add_argument("--trd",      type=int, default=0, help="TRD ID to train")
    parser.add_argument("--epochs",   type=int, default=1)
    parser.add_argument("--max-sents", type=int, default=1_000, dest="max_sentences")
    parser.add_argument("--passage-chars", type=int, default=TARGET_PASSAGE_CHARS, dest="passage_chars")
    parser.add_argument("--binary",   type=Path, default=Path("target/debug/csrre"))
    parser.add_argument("--bootstrap-dir", type=Path, default=None, dest="bootstrap_dir")
    parser.add_argument("--online-lexicon-update", action="store_true", dest="online_lexicon_update")
    parser.add_argument("--online-trd-update",     action="store_true", dest="online_trd_update")
    args = parser.parse_args(argv)

    config = TrainingConfig(
        language=args.lang,
        trd=args.trd,
        epochs=args.epochs,
        max_sentences=args.max_sentences,
        passage_chars=args.passage_chars,
        binary=args.binary,
        bootstrap_dir=args.bootstrap_dir,
        online_lexicon_update=args.online_lexicon_update,
        online_trd_update=args.online_trd_update,
    )

    trainer = SequentialTrainer(config)
    stats_list = trainer.train()

    if stats_list:
        final = stats_list[-1]
        print(f"Training complete: {final.passages} passages, {final.sentences} sentences, "
              f"mean quality = {final.mean_quality:.4f}")
    else:
        print("Training produced no statistics (check binary path and data).")


if __name__ == "__main__":
    main()
