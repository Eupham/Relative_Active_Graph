"""Subprocess bridge to the CSRRE Rust CLI over NDJSON."""
from __future__ import annotations
import json, logging, subprocess
from pathlib import Path
from subprocess import PIPE

logger = logging.getLogger(__name__)


class RustBridge:
    def __init__(self, binary: Path = Path("target/debug/csrre")) -> None:
        self._binary = binary
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "RustBridge":
        self._proc = subprocess.Popen(
            [str(self._binary)],
            stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True,
        )
        return self

    def __exit__(self, *_) -> None:
        if self._proc:
            try:
                self._proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

    def _send(self, msg: dict) -> None:
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        line = self._proc.stdout.readline().strip()
        if not line:
            return {}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            logger.error("Bad JSON from engine: %s", line)
            return {}

    def query(self, text: str, situation_id: int, trd: int | None = None,
              language: str = "en", nodes: list = (), edges: list = ()) -> dict:
        self._send({"type": "query", "text": text, "situation_id": situation_id,
                    "trd": trd, "language": language, "nodes": list(nodes), "edges": list(edges)})
        return self._recv()

    def register_lexicon(self, predicate: str, language: str, surface: str) -> None:
        self._send({"type": "register_lexicon", "predicate": predicate,
                    "language": language, "surface": surface})

    def execute_passage(self, trd: int, language: str, sentences: list[dict]) -> dict:
        """
        Submit a passage for teacher-forcing.
        sentences: list of {"nodes": [...], "edges": [...], "steps": [int, ...]}
        Returns: {"steps": int, "quality": float, "mean_quality": float}
        """
        self._send({"type": "execute_passage", "trd": trd, "language": language, "sentences": sentences})
        return self._recv()

    def make_node(self, node_id: int, surface: str, score: float = 0.5,
                  deprel_hash: int = 0, upos_hash: int = 0, arity: int = 0,
                  mode: str = "diamond", cat: int = 0) -> dict:
        return {"id": node_id, "surface": surface, "score": score,
                "deprel_hash": deprel_hash, "upos_hash": upos_hash,
                "arity": arity, "mode": mode, "cat": cat}

    def make_edge(self, edge_id: int, src: int, dst: int,
                  mode: str = "diamond", weight: float = 1.0) -> dict:
        return {"id": edge_id, "src": src, "dst": dst, "mode": mode, "weight": weight}
