# PRD: CSRRE Training Dashboard — Mission Control

## Original Problem Statement
1. Fix 6 code bugs in the Relative_Active_Graph repo (ATMS bit leak, LSE regression, fire-and-forget, FNV hash, converter, tokenizer)
2. Remove symbolica references — consolidate math to numerica_adapter, hashing to token_types
3. Build a GUI for end-to-end training with Teacher Forcing (Target-Constrained Derivation), ATMS-backed SCM Attribution, adaptive Poisson noise, and inference/generation

## Architecture
- **Rust Engine**: CSRRE core (ATMS, ARG, LCS, generation, semantics) compiled to `/app/target/release/csrre`
- **Python Training Pipeline**: `lcs/training/` — orchestrates C4 streaming + Rust bridge
- **Backend**: FastAPI (`/app/backend/server.py`) with WebSocket for live metrics
- **Frontend**: React + Tailwind (`/app/frontend/`) — "Neurosymbolic Mission Control" dashboard

## What's Been Implemented (Jan 2026)

### Phase 1: Bug Fixes (6 issues)
1. ContextStack::shift() ATMS bit leak — Added try_reclaim_pending between pop/push
2. normalize_path_weight — Restored LSE from geometric mean
3. register_lexicon — Changed fire-and-forget to self.send()
4. tokenizer.py — Updated _stable_node_id to 64-bit FNV-1a
5. FNV-1a consistency — All files using xor-then-multiply
6. sentence_to_mtlg — Already correctly calls inducer

### Phase 2: Numerica/Graphica Consolidation
- Removed all symbolica references
- numerica_adapter owns: normalize_path_weight, softmax, compute_depth, sample_discrete_noise
- token_types owns all FNV hashing via fnv1a_64_bytes
- Eliminated 7 inline FNV reimplementations, fixed 4 FNV-1 bugs

### Phase 3: Training Dashboard GUI
- **Training Config**: Language selector (99 mC4 languages), epochs, max sentences, passage chars
- **Start/Stop Training**: Spawns background thread orchestrating Rust engine + C4 streaming
- **Real-time Metrics**: Quality chart (Recharts), progress bar, epoch/passage/sentence/step counters
- **Adaptive Poisson Controller**: EMA-guided lambda auto-tuning
  - Success rate > 75% → increase noise (harder)
  - Success rate < 35% → decrease noise (easier)
  - Goldilocks zone → proportional fine-tuning
  - Lambda bounded [0.05, 5.0]
- **Simple/Advanced Toggle**: Simple shows quality + progress; Advanced adds Poisson panel, Engine State, Architecture Pipeline
- **Inference Panel**: Free generation with seed text, synonym query
- **System Log**: Terminal-style event feed with color-coded levels
- **WebSocket**: Live state broadcasting to all connected clients

### Testing Status
- 132/132 Rust tests pass
- Backend: 6/8 endpoints passing (inference timeouts handled gracefully with user message)
- Frontend: 100% all components functional
- Python hash consistency verified

## Backlog
- P0: Actual C4 training run end-to-end test (requires HuggingFace access)
- P1: Engine state save/load UI (engine.save/load already in Rust)
- P1: Add numerica unit tests for softmax and compute_depth
- P2: Derivation tree visualization in advanced view
- P2: Multi-language training support in UI
- P3: Export training metrics to CSV/JSON
