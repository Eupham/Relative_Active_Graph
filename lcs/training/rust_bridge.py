"""
rust_bridge.py — Python ↔ CSRRE Rust engine bridge.

Manages a subprocess running the `csrre` CLI binary (newline-delimited JSON
protocol) and exposes a Python API for:
  - Submitting queries
  - Running teacher-forcing training sequences
  - Registering lexicon entries
"""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterator, Optional


DEFAULT_BINARY = Path(__file__).parent.parent.parent / "target" / "debug" / "csrre"


class RustBridge:
    """Manages a long-lived CSRRE engine subprocess."""

    def __init__(self, binary: Path = DEFAULT_BINARY, env: Optional[dict] = None):
        self._binary = str(binary)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._env = env

    def start(self) -> None:
        """Launch the engine subprocess."""
        import os
        proc_env = os.environ.copy()
        if self._env:
            proc_env.update(self._env)
        self._proc = subprocess.Popen(
            [self._binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proc_env,
            text=True,
            bufsize=1,
        )

    def stop(self) -> None:
        """Send shutdown and wait for process to exit."""
        if self._proc is None:
            return
        self._send({"type": "shutdown"})
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def __enter__(self) -> "RustBridge":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ── Low-level I/O ─────────────────────────────────────────────────────────

    def _send(self, msg: dict) -> None:
        if self._proc is None:
            raise RuntimeError("RustBridge not started; call start() first")
        line = json.dumps(msg) + "\n"
        with self._lock:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

    def _recv(self) -> dict:
        if self._proc is None:
            raise RuntimeError("RustBridge not started")
        line = self._proc.stdout.readline()
        if not line:
            raise EOFError("Engine process closed stdout unexpectedly")
        return json.loads(line.strip())

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        situation_id: int = 1,
        trd: Optional[int] = None,
        language: str = "en",
        nodes: list[dict] = (),
        edges: list[dict] = (),
    ) -> dict:
        """Submit a query and return the result dict."""
        self._send({
            "type": "query",
            "text": text,
            "situation_id": situation_id,
            "trd": trd,
            "language": language,
            "nodes": list(nodes),
            "edges": list(edges),
        })
        return self._recv()

    def register_lexicon(
        self,
        predicate:  str,
        language:   str,
        surface:    str,
        role_order: list[str] | None = None,
    ) -> None:
        """Register a lexicon entry in the engine."""
        msg: dict = {
            "type":      "register_lexicon",
            "predicate": predicate,
            "language":  language,
            "surface":   surface,
        }
        if role_order:
            msg["role_order"] = role_order
        self._send(msg)

    def execute_passage(
        self,
        trd: int,
        language: str,
        sentences: list[dict],
    ) -> dict:
        """
        Submit a full passage as a teacher-forcing training unit.

        `sentences`: list of dicts, each with:
          - "nodes":  list of WireNode dicts
          - "edges":  list of WireEdge dicts
          - "steps":  list of expected_node_id ints (one per token)
        """
        self._send({
            "type":      "execute_passage",
            "trd":       trd,
            "language":  language,
            "sentences": sentences,
        })
        return self._recv()

    def make_node(
        self,
        node_id: int,
        surface: str,
        score: float = 0.5,
        deprel_hash: int = 0,
        upos_hash: int = 0,
        arity: int = 0,
        mode: str = "diamond",
    ) -> dict:
        """Build a WireNode dict. No hard category label is included."""
        return {
            "id":          node_id,
            "surface":     surface,
            "score":       score,
            "deprel_hash": deprel_hash,
            "upos_hash":   upos_hash,
            "arity":       arity,
            "mode":        mode,
        }

    def make_edge(
        self,
        edge_id: int,
        src: int,
        dst: int,
        mode: str = "diamond",
        weight: float = 1.0,
    ) -> dict:
        """Build a WireEdge dict."""
        return {"id": edge_id, "src": src, "dst": dst, "mode": mode, "weight": weight}
