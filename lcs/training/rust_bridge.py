"""Subprocess bridge to the CSRRE Rust CLI over NDJSON."""
from __future__ import annotations
import json, logging, subprocess, threading, select, time
from pathlib import Path

logger = logging.getLogger(__name__)


class RustBridge:
    READ_TIMEOUT = 30.0  # seconds

    def __init__(self, binary: Path = Path("target/debug/csrre"),
                 state_path: str = "") -> None:
        self._binary     = str(binary)
        self._state_path = state_path
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self._launch()

    def stop(self) -> None:
        with self._lock:
            if self._proc:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
                self._proc = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def ensure_running(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self.restart()

    def _launch(self) -> None:
        cmd = [self._binary]
        if self._state_path:
            cmd += ["--load-state", self._state_path]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        logging.info("Rust engine started (pid=%d)", self._proc.pid)

    def _drain_stderr(self) -> None:
        for raw in self._proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            logging.error("Rust[stderr]: %s", line)
            if "panicked" in line:
                logging.critical("Rust panic — bridge will restart on next call")

    def send(self, payload: str) -> str:
        self.ensure_running()
        self._proc.stdin.write((payload + "\n").encode())
        self._proc.stdin.flush()
        ready, _, _ = select.select([self._proc.stdout], [], [], self.READ_TIMEOUT)
        if not ready:
            logging.error("Rust bridge read timeout — restarting")
            self.restart()
            raise TimeoutError("Rust engine did not respond within timeout")
        line = self._proc.stdout.readline()
        if not line:
            self.restart()
            raise BrokenPipeError("Rust engine closed stdout")
        return line.decode("utf-8").rstrip()

    # ── Legacy context manager API (backward compat) ──────────────────────────

    def __enter__(self) -> "RustBridge":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def _send_json(self, msg: dict) -> None:
        self.send(json.dumps(msg))

    def _recv_json(self) -> dict:
        try:
            line = self.send("")
            return json.loads(line) if line else {}
        except Exception:
            return {}

    def query(self, text: str, situation_id: int, trd: int | None = None,
              language: str = "en", nodes: list = (), edges: list = ()) -> dict:
        payload = json.dumps({"type": "query", "text": text,
                              "situation_id": situation_id, "trd": trd,
                              "language": language, "nodes": list(nodes),
                              "edges": list(edges)})
        result = self.send(payload)
        try:
            return json.loads(result)
        except Exception:
            return {}

    def register_lexicon(self, predicate: str, language: str, surface: str) -> None:
        payload = json.dumps({"type": "register_lexicon", "predicate": predicate,
                              "language": language, "surface": surface})
        self.send(payload)

    def execute_passage(self, trd: int, language: str, sentences: list[dict]) -> dict:
        payload = json.dumps({"type": "execute_passage", "trd": trd,
                              "language": language, "sentences": sentences})
        result = self.send(payload)
        try:
            return json.loads(result)
        except Exception:
            return {}

    def make_node(self, node_id: int, surface: str, score: float = 0.5,
                  deprel_hash: int = 0, upos_hash: int = 0, arity: int = 0,
                  mode: str = "diamond", cat: int = 0) -> dict:
        return {"id": node_id, "surface": surface, "score": score,
                "deprel_hash": deprel_hash, "upos_hash": upos_hash,
                "arity": arity, "mode": mode, "cat": cat}

    def make_edge(self, edge_id: int, src: int, dst: int,
                  mode: str = "diamond", weight: float = 1.0) -> dict:
        return {"id": edge_id, "src": src, "dst": dst, "mode": mode, "weight": weight}
