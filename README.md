# Relative Active Graph (RAG)

A **Constraint-Scheduled Reactive Reasoning Engine** implementing multimodal
type-logical semantic composition with adaptive learning.

The system builds a graph representation of natural language using modal type logic
(MTLG), maintains contextual consistency via an ATMS, and learns edge-weight
associations from streaming text (C4). Discourse referents are tracked in
DRS-structured context frames. Category labels follow UCCA/AMR naming conventions;
full compliance with those formalisms is a development target, not a current
capability.

---

## App Launcher — Notebook

`Launch.ipynb` starts the Colab/local launcher script (`colab_launch.py`) which
builds the React dashboard, starts FastAPI on port 8000, and exposes it via
LocalTunnel.

Run it locally:

```bash
pip install notebook
jupyter notebook Launch.ipynb
```

The notebook is structured as three cells — run them in order:

| Cell | What it does |
|------|-------------|
| 1 | Install Python deps (`lcs/requirements.txt`, `fastapi`, `uvicorn`) |
| 2 | Build Rust engine via `cargo build --release` |
| 3 | Execute `colab_launch.py` (installs npm deps, builds React, starts FastAPI on `:8000`, opens LocalTunnel) |

A final optional cell sends SIGTERM to all started processes.

**Note:** The launcher is designed for local notebooks and Colab-style runtimes.
If `npm run build` fails due stale dependencies, `colab_launch.py` retries with a
clean install automatically.
If you run npm manually, do it inside `frontend/` (not repo root), e.g.
`cd frontend && npm install`.

---

## Streamlit UI

Run the interactive web interface locally:

```bash
pip install -r lcs/requirements.txt
streamlit run app.py
```

The UI provides two tabs:
- **Train** — stream C4, build lexicon and TRD clusters with live progress
- **Inference & Visualization** — tokenize a sentence, render its MTLG graph, and query the Rust engine for a surface output

---

## Integration Tests

Quick pipeline check (pure Python, no Rust required):

```bash
cd lcs/induction
python test_pipeline.py
```

Full end-to-end IPC path test (requires Rust build):

```bash
cargo build --release
python test_e2e_english.py "Alice discovered a particle."
```

Expected output is a single token (`Alice`). This confirms the Python → Rust subprocess
path is functional. Multi-token generation requires a trained lexicon loaded at engine
startup; see `test_summary.md` for current status.

---

## Repository Layout

```
Relative_Active_Graph/
├── Launch.ipynb                   # One-click launcher — installs deps, builds Rust, starts full stack
├── app.py                         # Streamlit web UI (Train + Inference & Visualization)
├── test_e2e_english.py            # End-to-end English pipeline test (Rust + Python)
├── test_summary.md                # Test execution results
├── core/                          # Rust reasoning engine (CSRRE)
│   └── src/
│       ├── engine.rs              # 13-step main reasoning loop
│       ├── arg/                   # Active Relative Graph (lazy, budget-bounded)
│       ├── atms/                  # Base ATMS + BF-ATMS (counterfactuals)
│       ├── adaptive/              # VAT (Variance-Adaptive Thresholds) + EMA performance
│       ├── scheduler/             # b/t-level list scheduling + critical path
│       ├── constraints/           # SHACL shapes + sheaf coherence + Z3
│       ├── causal/                # SCM + interventions + counterfactuals
│       ├── semantics/             # MTLG semantics (lambda/proposition + DRS referent sets;
│       │                          #   UCCA/AMR naming used for categories, full parsers not included)
│       ├── generation/            # Progressive deepening + lineariser
│       ├── feedback/              # Attribution + provenance + edge updates
│       ├── rules/                 # Rule induction lifecycle (Ruler)
│       └── lcs/                   # Language category system
├── cli/                           # Newline-delimited JSON query interface
└── lcs/
    ├── requirements.txt           # Python dependencies
    ├── induction/                 # Python induction pipeline
    │   ├── mc4_stream.py          # C4 / mC4 streaming (108 languages)
    │   ├── boundary_inducer.py    # Grammar-driven character-level boundary induction
    │   ├── community_inducer.py   # SBM Leiden MDL community detection
    │   ├── morphological_fst.py   # FST decomposition (Turkish, Finnish, …)
    │   ├── mtlg_inducer.py        # Probabilistic lexicon induction (MLE)
    │   ├── lexicon_inducer.py     # SuffixParadigmLattice (Goldsmith 2001)
    │   ├── hol_to_lc.py           # HOL ↔ λ-calculus conversion
    │   └── test_pipeline.py       # Integration test
    └── training/                  # Rust-Python training bridges
        ├── sequential_trainer.py  # Sequential edge-weight training on C4
        ├── bootstrap_trainer.py   # Phase 0: bootstrap artefacts
        ├── c4_sequence_extractor.py
        └── rust_bridge.py         # Subprocess NDJSON bridge
```

## Rust Build

```bash
cargo build --release
```

## Python Induction Pipeline

```bash
pip install -r lcs/requirements.txt
python lcs/induction/mtlg_inducer.py en    # induce English lexicon from C4
python lcs/induction/trd_bootstrap.py en   # bootstrap TRDs
```
