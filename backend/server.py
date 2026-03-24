"""CSRRE Training Dashboard — FastAPI backend.
Orchestrates the Rust engine, streams training metrics via WebSocket,
and serves inference endpoints.
"""
from __future__ import annotations
import asyncio
import json
import logging
import math
import os
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "lcs" / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent / "lcs" / "induction"))

app = FastAPI(title="CSRRE Training Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    build_index = Path(__file__).parent.parent / "frontend" / "build" / "index.html"
    if build_index.exists():
        return FileResponse(str(build_index))
    return {
        "message": "CSRRE backend online. Frontend build not found; run `npm run build` in frontend/.",
        "status": "online",
        "docs_url": "/docs",
    }

RUST_BINARY = Path(os.environ.get("RUST_BINARY", "/app/target/release/csrre"))

# ── Adaptive Poisson Controller ───────────────────────────────────────────────

class AdaptivePoissonController:
    """
    Dynamically adjusts the Poisson lambda for rule-dropping noise.
    
    Strategy: Maintains an exponential moving average of derivation success rate.
    - If success_rate > 0.75: increase lambda (harder) — system is too comfortable
    - If success_rate < 0.35: decrease lambda (easier) — system is drowning
    - Otherwise: fine-tune proportionally
    
    Bounds: lambda in [0.05, 5.0]
    """
    def __init__(self):
        self.lam = 0.5
        self.min_lam = 0.05
        self.max_lam = 5.0
        self.ema_success = 0.5
        self.alpha = 0.1  # EMA smoothing
        self.history = []

    def update(self, quality: float, success: bool):
        self.ema_success = self.alpha * (1.0 if success else 0.0) + (1 - self.alpha) * self.ema_success
        
        if self.ema_success > 0.75:
            # Too easy — increase noise pressure
            self.lam = min(self.max_lam, self.lam * 1.08)
        elif self.ema_success < 0.35:
            # Drowning — ease up
            self.lam = max(self.min_lam, self.lam * 0.92)
        else:
            # Goldilocks zone — gentle proportional adjustment
            target = 0.55
            error = self.ema_success - target
            self.lam = max(self.min_lam, min(self.max_lam, self.lam * (1 + error * 0.05)))
        
        self.history.append({
            "lambda": round(self.lam, 4),
            "ema_success": round(self.ema_success, 4),
            "quality": round(quality, 4),
        })

    def snapshot(self):
        return {
            "lambda": round(self.lam, 4),
            "ema_success": round(self.ema_success, 4),
            "min_lam": self.min_lam,
            "max_lam": self.max_lam,
            "history_len": len(self.history),
        }

# ── Training State ────────────────────────────────────────────────────────────

@dataclass
class TrainingState:
    running: bool = False
    epoch: int = 0
    total_epochs: int = 1
    passages: int = 0
    sentences: int = 0
    steps: int = 0
    quality_sum: float = 0.0
    mean_quality: float = 0.0
    rules_induced: int = 0
    global_nodes: int = 0
    global_edges: int = 0
    events: list = field(default_factory=list)
    quality_history: list = field(default_factory=list)
    poisson: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    max_sentences: int = 1000
    language: str = "en"
    dataset: str = "c4"
    error: str = ""

    def to_dict(self):
        return {
            "running": self.running,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "passages": self.passages,
            "sentences": self.sentences,
            "steps": self.steps,
            "mean_quality": round(self.mean_quality, 4),
            "rules_induced": self.rules_induced,
            "global_nodes": self.global_nodes,
            "global_edges": self.global_edges,
            "events": self.events[-100:],
            "quality_history": self.quality_history[-500:],
            "poisson": self.poisson,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "max_sentences": self.max_sentences,
            "language": self.language,
            "dataset": self.dataset,
            "error": self.error,
        }

training_state = TrainingState()
poisson_ctrl = AdaptivePoissonController()
ws_clients: list[WebSocket] = []
_stop_event = threading.Event()
_training_thread: Optional[threading.Thread] = None

# ── Broadcast to WebSocket clients ────────────────────────────────────────────

async def broadcast(msg: dict):
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        ws_clients.remove(d)

def sync_broadcast(msg: dict):
    """Thread-safe broadcast from training thread."""
    loop = getattr(sync_broadcast, '_loop', None)
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(msg), loop)

# ── Training Thread ───────────────────────────────────────────────────────────

def _run_training(config: dict):
    global training_state, poisson_ctrl
    
    training_state.running = True
    training_state.error = ""
    training_state.events = []
    training_state.quality_history = []
    training_state.passages = 0
    training_state.sentences = 0
    training_state.steps = 0
    training_state.quality_sum = 0.0
    training_state.mean_quality = 0.0
    poisson_ctrl = AdaptivePoissonController()
    
    language = config.get("language", "en")
    dataset = (config.get("dataset", "c4") or "c4").lower()
    if dataset != "c4":
        dataset = "c4"
    epochs = config.get("epochs", 1)
    max_sentences = config.get("max_sentences", 1000)
    passage_chars = config.get("passage_chars", 2000)
    
    training_state.total_epochs = epochs
    training_state.max_sentences = max_sentences
    training_state.language = language
    training_state.dataset = dataset
    
    start_time = time.time()
    
    try:
        from rust_bridge import RustBridge
        from c4_sequence_extractor import stream_c4_sequences, extract_sequence
        
        binary = RUST_BINARY
        if not binary.exists():
            debug_bin = Path("/app/target/debug/csrre")
            binary = debug_bin if debug_bin.exists() else binary
        
        if not binary.exists():
            training_state.error = f"Rust binary not found: {binary}"
            training_state.running = False
            _emit_event("error", training_state.error)
            return
        
        _emit_event("info", f"Starting engine: {binary}")
        
        with RustBridge(binary=binary) as bridge:
            _emit_event("info", f"Engine started. Training {language} on {dataset.upper()} — {epochs} epoch(s), max {max_sentences} sentences")
            
            for epoch in range(epochs):
                if _stop_event.is_set():
                    _emit_event("info", "Training stopped by user")
                    break
                
                training_state.epoch = epoch + 1
                _emit_event("info", f"Epoch {epoch + 1}/{epochs} starting")
                
                passage_buffer = []
                passage_char_count = 0
                
                from tokenizer import _stable_node_id, _fnv_hash, contextual_node_id, infer_modal_mode
                
                def flush_passage():
                    nonlocal passage_char_count
                    if not passage_buffer:
                        return
                    
                    wire_sentences = []
                    for seq in passage_buffer:
                        nodes = []
                        for s in seq.steps:
                            # Infer ModalMode from structural features
                            n_toks = len(seq.steps)
                            pos_idx = seq.steps.index(s)
                            is_first = (pos_idx == 0)
                            is_last = (pos_idx == n_toks - 1)
                            norm_pos = pos_idx / max(n_toks - 1, 1)
                            is_repeated = sum(1 for x in seq.steps if x.lemma == s.lemma) > 1
                            mode_int = infer_modal_mode(is_repeated, norm_pos, is_last)
                            mode_str = ["diamond", "box", "lozenge"][mode_int]
                            
                            # Use contextual_node_id for IPC parity with Rust
                            nid = contextual_node_id(s.suffix3_hash, s.prefix2_hash, 0, mode_int)
                            
                            nodes.append(bridge.make_node(
                                node_id=nid,
                                surface=s.text,
                                score=0.5,
                                deprel_hash=s.suffix3_hash,
                                upos_hash=s.prefix2_hash,
                                arity=0,
                                mode=mode_str,
                                cat=0,
                            ))
                        edges = [
                            bridge.make_edge(
                                edge_id=(i + 1),
                                src=contextual_node_id(
                                    seq.steps[i].suffix3_hash, seq.steps[i].prefix2_hash,
                                    0, infer_modal_mode(
                                        sum(1 for x in seq.steps if x.lemma == seq.steps[i].lemma) > 1,
                                        i / max(len(seq.steps) - 1, 1),
                                        i == len(seq.steps) - 1,
                                    )
                                ),
                                dst=contextual_node_id(
                                    seq.steps[i + 1].suffix3_hash, seq.steps[i + 1].prefix2_hash,
                                    0, infer_modal_mode(
                                        sum(1 for x in seq.steps if x.lemma == seq.steps[i + 1].lemma) > 1,
                                        (i + 1) / max(len(seq.steps) - 1, 1),
                                        (i + 1) == len(seq.steps) - 1,
                                    )
                                ),
                            )
                            for i in range(len(seq.steps) - 1)
                        ]
                        # Build step IDs with contextual hashing
                        step_ids = []
                        for idx_s, s in enumerate(seq.steps):
                            is_rep = sum(1 for x in seq.steps if x.lemma == s.lemma) > 1
                            npos = idx_s / max(len(seq.steps) - 1, 1)
                            is_lst = idx_s == len(seq.steps) - 1
                            step_ids.append(contextual_node_id(
                                s.suffix3_hash, s.prefix2_hash, 0,
                                infer_modal_mode(is_rep, npos, is_lst)
                            ))
                        wire_sentences.append({
                            "nodes": nodes, "edges": edges,
                            "steps": step_ids,
                        })
                    
                    # Register lexicon
                    registered = set()
                    for sent in wire_sentences:
                        for node in sent["nodes"]:
                            surface = node["surface"]
                            lemma = surface.lower()
                            if lemma not in registered:
                                bridge.register_lexicon(predicate=lemma, language=language, surface=surface)
                                registered.add(lemma)
                    
                    # Execute passage (teacher forcing)
                    result = bridge.execute_passage(trd=0, language=language, sentences=wire_sentences)
                    
                    p_steps = int(result.get("steps", 0))
                    p_quality = float(result.get("mean_quality", 0.0))
                    success = p_quality >= 0.3
                    
                    training_state.passages += 1
                    training_state.sentences += len(passage_buffer)
                    training_state.steps += p_steps
                    training_state.quality_sum += p_quality * max(p_steps, 1)
                    training_state.mean_quality = training_state.quality_sum / max(training_state.steps, 1)
                    training_state.elapsed_sec = time.time() - start_time
                    
                    # Update adaptive Poisson
                    poisson_ctrl.update(p_quality, success)
                    training_state.poisson = poisson_ctrl.snapshot()
                    
                    # Track quality history
                    training_state.quality_history.append({
                        "passage": training_state.passages,
                        "quality": round(p_quality, 4),
                        "lambda": round(poisson_ctrl.lam, 4),
                        "ema_success": round(poisson_ctrl.ema_success, 4),
                    })
                    
                    # Track engine stats from result
                    training_state.global_nodes = int(result.get("global_nodes", training_state.global_nodes))
                    training_state.global_edges = int(result.get("global_edges", training_state.global_edges))
                    training_state.rules_induced = int(result.get("rules_count", training_state.rules_induced))
                    
                    if training_state.passages % 10 == 0:
                        _emit_event("metric", 
                            f"P{training_state.passages}: quality={p_quality:.4f} lambda={poisson_ctrl.lam:.3f} "
                            f"ema_success={poisson_ctrl.ema_success:.3f}")
                    
                    sync_broadcast({"type": "state", "data": training_state.to_dict()})
                    
                    passage_buffer.clear()
                    passage_char_count = 0
                
                # Stream C4 data
                try:
                    for seq in stream_c4_sequences(language=language, max_sentences=max_sentences):
                        if _stop_event.is_set():
                            break
                        passage_buffer.append(seq)
                        passage_char_count += len(seq.sentence)
                        if passage_char_count >= passage_chars:
                            flush_passage()
                    
                    flush_passage()  # Flush remaining
                except Exception as e:
                    _emit_event("error", f"C4 streaming error: {e}")
                    logger.exception("C4 error")
                
                _emit_event("info", f"Epoch {epoch + 1} complete: {training_state.passages} passages, quality={training_state.mean_quality:.4f}")
        
        _emit_event("info", f"Training complete! {training_state.passages} passages, mean_quality={training_state.mean_quality:.4f}")
    
    except Exception as e:
        training_state.error = str(e)
        _emit_event("error", f"Training failed: {e}")
        logger.exception("Training error")
    finally:
        training_state.running = False
        sync_broadcast({"type": "state", "data": training_state.to_dict()})


def _emit_event(level: str, message: str):
    evt = {"ts": time.time(), "level": level, "message": message}
    training_state.events.append(evt)
    sync_broadcast({"type": "event", "data": evt})
    if level == "error":
        logger.error(message)
    else:
        logger.info(message)

# ── Pydantic Models ───────────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    dataset: str = "c4"
    language: str = "en"
    epochs: int = 1
    max_sentences: int = 1000
    passage_chars: int = 2000

class InferenceRequest(BaseModel):
    seed_text: str
    max_tokens: int = 64
    language: str = "en"

class SynonymRequest(BaseModel):
    word: str
    n: int = 5

# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "engine_binary": str(RUST_BINARY), "binary_exists": RUST_BINARY.exists()}

@app.post("/api/training/start")
def start_training(config: TrainConfig):
    global _training_thread
    if training_state.running:
        return {"error": "Training already running"}
    _stop_event.clear()
    _training_thread = threading.Thread(target=_run_training, args=(config.dict(),), daemon=True)
    _training_thread.start()
    return {"status": "started", "config": config.dict()}

@app.post("/api/training/stop")
def stop_training():
    if not training_state.running:
        return {"error": "Training not running"}
    _stop_event.set()
    return {"status": "stopping"}

@app.get("/api/training/status")
def get_status():
    return training_state.to_dict()

@app.get("/api/training/poisson")
def get_poisson():
    return {
        "controller": poisson_ctrl.snapshot(),
        "history": poisson_ctrl.history[-200:],
    }

@app.post("/api/inference/generate")
def generate(req: InferenceRequest):
    """Run free generation through the engine."""
    try:
        from rust_bridge import RustBridge
        
        binary = RUST_BINARY
        if not binary.exists():
            return {"error": f"Binary not found: {binary}"}
        
        bridge = RustBridge(binary=binary)
        bridge.READ_TIMEOUT = 15.0
        bridge.start()
        
        try:
            # Register seed words
            words = req.seed_text.split()
            def _stable_node_id(lemma: str) -> int:
                h = 0xcbf29ce484222325
                for b in lemma.encode():
                    h = ((h ^ b) * 0x00000100000001b3) & 0xFFFFFFFFFFFFFFFF
                return h
            
            for w in words:
                bridge.register_lexicon(predicate=w.lower(), language=req.language, surface=w)
            
            result = bridge.query(
                text=req.seed_text,
                situation_id=1,
                trd=0,
                language=req.language,
                nodes=[bridge.make_node(node_id=_stable_node_id(w.lower()), surface=w, score=0.8) for w in words],
                edges=[],
            )
            
            return {
                "output": result.get("surface_output", ""),
                "quality": result.get("quality", 0),
                "satisfied": result.get("satisfied", False),
                "depth_used": result.get("depth_used", 0),
            }
        finally:
            bridge.stop()
    except TimeoutError:
        return {"error": "Engine timed out. Train the model first to build knowledge.", "output": ""}
    except Exception as e:
        return {"error": str(e), "output": ""}

@app.post("/api/inference/synonym")
def synonym_query(req: SynonymRequest):
    """Query synonyms from the engine's global graph."""
    try:
        from rust_bridge import RustBridge
        binary = RUST_BINARY
        if not binary.exists():
            return {"error": f"Binary not found: {binary}"}
        
        bridge = RustBridge(binary=binary)
        bridge.READ_TIMEOUT = 10.0
        bridge.start()
        
        try:
            bridge.register_lexicon(predicate=req.word.lower(), language="en", surface=req.word)
            result = bridge.query(text=req.word, situation_id=1, language="en")
            return {"word": req.word, "result": result}
        finally:
            bridge.stop()
    except TimeoutError:
        return {"error": "Engine timed out. Train the model first.", "word": req.word, "result": {}}
    except Exception as e:
        return {"error": str(e), "word": req.word, "result": {}}

@app.get("/api/engine/info")
def engine_info():
    """Engine system information."""
    binary_exists = RUST_BINARY.exists()
    return {
        "binary_path": str(RUST_BINARY),
        "binary_exists": binary_exists,
        "languages_supported": [
            "af","am","ar","az","be","bg","bn","ca","cs","cy","da","de","el","en","eo","es",
            "et","eu","fa","fi","fr","fy","ga","gl","gu","ha","hi","hr","ht","hu","hy","id",
            "ig","is","it","iw","ja","ka","kk","km","kn","ko","ku","ky","la","lb","lo","lt",
            "lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","no","ny","pa","pl",
            "ps","pt","ro","ru","sd","si","sk","sl","sm","sn","so","sq","sr","st","su","sv",
            "sw","ta","te","tg","th","tk","tl","tr","tt","ug","uk","ur","uz","vi","xh","yi",
            "yo","zh","zu",
        ],
    }

# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/api/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    sync_broadcast._loop = asyncio.get_event_loop()
    try:
        # Send current state immediately
        await websocket.send_json({"type": "state", "data": training_state.to_dict()})
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)


# ── Serve React Build ─────────────────────────────────────────────
# After all API routes so the catch-all doesn't swallow /api/* paths.

import pathlib as _pathlib
_BUILD = _pathlib.Path(__file__).parent.parent / "frontend" / "build"

if _BUILD.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse as _FileResponse

    app.mount(
        "/static",
        StaticFiles(directory=_BUILD / "static"),
        name="react-static",
    )

    # Serve any other path as the SPA index (React Router handles the rest)
    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        index = _BUILD / "index.html"
        return _FileResponse(str(index))
else:
    logger.warning(
        "React build not found at %s — run `npm run build` inside frontend/",
        _BUILD,
    )
