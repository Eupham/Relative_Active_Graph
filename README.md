# Relative Active Graph (RAG)

A **Constraint-Scheduled Reactive Reasoning Engine** implementing multimodal
type-logical semantic composition with adaptive learning.

The system grounds natural language in formal semantics (MTLG / DRS / UCCA / AMR)
and uses situation-relative reasoning with ATMS-backed truth maintenance,
causal SCMs, and Variance-Adaptive Thresholds (VAT).

---

## App Launcher — Notebook

`Launch.ipynb` starts the **full application stack in one go**: FastAPI backend,
React dashboard, and Streamlit UI (fallback when npm is unavailable).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eupham/Relative_Active_Graph/blob/master/Launch.ipynb)

### What the launcher does

| Step | Description |
|------|-------------|
| 1 | Clone repo from GitHub (Colab only — skipped when already present) |
| 2 | Install Python deps (`lcs/requirements.txt`, `fastapi`, `uvicorn`, `pyngrok`) |
| 3 | Build Rust engine via `cargo build --release` (skipped if binary exists or cargo unavailable) |
| 4 | Install React frontend deps (`npm install`) |
| 5 | Start FastAPI backend on port 8000, React dashboard on port 3000 (or Streamlit on 8501) |
| 6 | **Colab**: expose services via ngrok and print public URLs — **Local**: print `localhost` URLs |

A final optional cell shuts everything down cleanly.

---

## Streamlit UI

Run the interactive web interface locally:

```bash
pip install -r lcs/requirements.txt
streamlit run app.py
```

The UI provides two tabs:
- **Train** — stream C4, induce lexicon, bootstrap TRDs with live progress bars
- **Inference & Visualization** — analyse sentences and render MTLG dependency graphs

---

## Integration Tests

Quick pipeline check (pure Python, no Rust required):

```bash
cd lcs/induction
python test_pipeline.py
```

Full end-to-end test (requires Rust build):

```bash
cargo build --release
python test_e2e_english.py
```

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
│       ├── semantics/             # MTLG semantics (AMR + DRS + UCCA)
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
        ├── sequential_trainer.py  # Teacher-forcing on C4
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
