"""Bootstrap-free teacher-forcing trainer. No UD, no Stanza, no bootstrap phase."""
from __future__ import annotations
import argparse, hashlib, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TARGET_PASSAGE_CHARS = 2000
_MODE_TO_TRD = {"diamond": 0, "box": 1, "lozenge": 2}
UCCA_TO_INT  = {"Scene": 0, "Process": 1, "Connector": 2, "Ground": 3,
                "Adverbial": 4, "State": 5, "Participant": 6}


def _stable_node_id(lemma: str) -> int:
    return int(hashlib.sha256(lemma.encode()).hexdigest(), 16) & 0xFFFFFFFFFFFF


def _fnv_hash(s: str) -> int:
    h = 0x811c9dc5
    for b in s.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def passage_trd(wire_sentences: list[dict]) -> int:
    counts: dict[str, int] = {"diamond": 0, "box": 0, "lozenge": 0}
    for sent in wire_sentences:
        for node in sent.get("nodes", []):
            m = node.get("mode", "diamond")
            counts[m] = counts.get(m, 0) + 1
    return _MODE_TO_TRD.get(max(counts, key=counts.get), 0)


@dataclass
class TrainingConfig:
    language:        str  = "en"
    epochs:          int  = 1
    max_sentences:   int  = 1_000
    log_interval:    int  = 100
    passage_chars:   int  = TARGET_PASSAGE_CHARS
    binary:          Path = field(default_factory=lambda: Path("target/debug/csrre"))
    category_warmup: int  = 100


@dataclass
class EpochStats:
    epoch:       int
    sentences:   int   = 0
    steps:       int   = 0
    quality_sum: float = 0.0
    passages:    int   = 0

    @property
    def mean_quality(self) -> float:
        return self.quality_sum / max(self.steps, 1)


class SequentialTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config   = config
        self._stats:  list[EpochStats] = []
        self._inducer = None
        self._passages_seen   = 0
        self._acc_graphs: list = []

    def train(self) -> list[EpochStats]:
        from .rust_bridge import RustBridge
        binary = self.config.binary
        if not binary.exists():
            release = binary.parent.parent / "release" / binary.name
            binary  = release if release.exists() else binary
        if not binary.exists():
            logger.error("Binary not found: %s", binary)
            return []
        with RustBridge(binary=binary) as bridge:
            for epoch in range(self.config.epochs):
                stats = self._run_epoch(bridge, epoch)
                self._stats.append(stats)
                logger.info("Epoch %d: passages=%d sentences=%d mean_quality=%.4f",
                            epoch, stats.passages, stats.sentences, stats.mean_quality)
        return self._stats

    def _run_epoch(self, bridge, epoch: int) -> EpochStats:
        from .c4_sequence_extractor import stream_c4_sequences

        stats          = EpochStats(epoch=epoch)
        passage_buffer: list = []
        passage_chars   = 0

        def flush_passage() -> None:
            nonlocal passage_chars
            if not passage_buffer:
                return

            wire_sentences = []
            for seq in passage_buffer:
                nodes = []
                for s in seq.steps:
                    cat_id = 0
                    if self._inducer is not None:
                        entry = self._inducer.best_type(s.lemma)
                        if entry is not None:
                            cat_id = UCCA_TO_INT.get(entry.ucca_cat, 0)
                    nodes.append(bridge.make_node(
                        node_id=_stable_node_id(s.lemma),
                        surface=s.text,
                        score=0.5,
                        deprel_hash=s.suffix3_hash,
                        upos_hash=s.prefix2_hash,
                        arity=0,
                        mode="diamond",
                        cat=cat_id,
                    ))
                edges = [
                    bridge.make_edge(
                        edge_id=(i + 1),
                        src=_stable_node_id(seq.steps[i].lemma),
                        dst=_stable_node_id(seq.steps[i + 1].lemma),
                    )
                    for i in range(len(seq.steps) - 1)
                ]
                wire_sentences.append({
                    "nodes": nodes, "edges": edges,
                    "steps": [_stable_node_id(s.lemma) for s in seq.steps],
                })

            registered: set = set()
            for sent in wire_sentences:
                for node in sent["nodes"]:
                    surface = node["surface"]
                    lemma   = surface.lower()
                    if lemma not in registered:
                        bridge.register_lexicon(predicate=lemma,
                                                language=self.config.language,
                                                surface=surface)
                        registered.add(lemma)

            trd    = passage_trd(wire_sentences)
            result = bridge.execute_passage(trd=trd, language=self.config.language,
                                            sentences=wire_sentences)

            stats.passages    += 1
            stats.sentences   += len(passage_buffer)
            stats.steps       += int(result.get("steps", 0))
            stats.quality_sum += float(result.get("mean_quality", 0.0)) * max(int(result.get("steps", 1)), 1)

            self._passages_seen += 1
            if self._passages_seen == self.config.category_warmup and self._inducer is None and self._acc_graphs:
                self._fit_inducer()

            passage_buffer.clear()
            passage_chars = 0

        for seq in stream_c4_sequences(language=self.config.language, max_sentences=self.config.max_sentences):
            if self._passages_seen < self.config.category_warmup and self._inducer is None:
                self._acc_graphs.append(seq)
            passage_buffer.append(seq)
            passage_chars += len(seq.sentence)
            if passage_chars >= self.config.passage_chars:
                flush_passage()
            if stats.passages > 0 and stats.passages % self.config.log_interval == 0:
                logger.info("  [epoch %d] passages=%d mean_quality=%.4f",
                            epoch, stats.passages, stats.mean_quality)

        flush_passage()
        return stats

    def _fit_inducer(self) -> None:
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "induction"))
            from text_to_graph import text_to_mtlg
            from mtlg_inducer import MtlgInducer

            self._inducer = MtlgInducer(self.config.language)
            sentences = [seq.sentence for seq in self._acc_graphs if seq.sentence]
            graphs    = []
            for sent in sentences[:500]:
                try:
                    g = text_to_mtlg(sent, self.config.language)
                    if g.nodes:
                        graphs.append(g)
                except Exception:
                    pass
            if graphs:
                self._inducer.induce_from_stream(iter(graphs), max_trees=len(graphs))
                logger.info("CategoryInducer fitted on %d graphs (%d lemmas)",
                            len(graphs), len(self._inducer.entries))
        except Exception as exc:
            logger.warning("CategoryInducer fit failed: %s", exc)
            self._inducer = None
        finally:
            self._acc_graphs.clear()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="CSRRE bootstrap-free trainer")
    parser.add_argument("--lang",             default="en")
    parser.add_argument("--epochs",           type=int, default=1)
    parser.add_argument("--max-sents",        type=int, default=1_000, dest="max_sentences")
    parser.add_argument("--passage-chars",    type=int, default=TARGET_PASSAGE_CHARS, dest="passage_chars")
    parser.add_argument("--binary",           type=Path, default=Path("target/debug/csrre"))
    parser.add_argument("--category-warmup",  type=int, default=100, dest="category_warmup")
    args = parser.parse_args(argv)

    config  = TrainingConfig(language=args.lang, epochs=args.epochs,
                             max_sentences=args.max_sentences, passage_chars=args.passage_chars,
                             binary=args.binary, category_warmup=args.category_warmup)
    trainer = SequentialTrainer(config)
    stats   = trainer.train()
    if stats:
        f = stats[-1]
        print(f"Training complete: {f.passages} passages, mean_quality={f.mean_quality:.4f}")
    else:
        print("No output. Check binary path.")


if __name__ == "__main__":
    main()
