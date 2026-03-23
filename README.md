# Relative Active Graph (RAG)

A **Constraint-Scheduled Reactive Reasoning Engine** implementing multimodal
type-logical semantic composition with adaptive learning.

The system grounds natural language in formal semantics (MTLG / DRS / UCCA / AMR)
and uses situation-relative reasoning with ATMS-backed truth maintenance,
causal SCMs, and Variance-Adaptive Thresholds (VAT).

---

## C4 Training — Colab Notebook

Train the MTLG lexicon and TRD bootstrapper on a streaming subset of the
[C4 corpus](https://huggingface.co/datasets/allenai/c4) (HuggingFace), run the
integration test suite, then launch the interactive **Streamlit UI** via a public
ngrok URL — no GPU required, ~5 min on a free Colab CPU.

The notebook **clones this repository directly** so it always runs against the
current codebase — no inline source copies.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eupham/Relative_Active_Graph/blob/master/RAG_C4_Training.ipynb)

### What the notebook covers

| Step | Description |
|------|-------------|
| 1 | **Clone repo from GitHub** — source modules loaded from the live codebase |
| 2 | Install dependencies (`datasets`, `leidenalg`, `python-igraph`, `numpy`, `scipy`, `networkx`, `streamlit`, `pyngrok`, …) |
| 3 | Add repo modules to path |
| 4 | Configure demo parameters (`TRAIN_SAMPLES`, `HELD_OUT`) |
| 5–8 | **Stream C4** → boundary induction → MTLG graphs → induce lexicon → bootstrap TRDs (SBM Leiden MDL) → evaluate held-out |
| 9 | **Integration test suite** — 5 canonical sentences verified end-to-end (parse, lexicon, TRD) |
| 10 | Visualise modal mode and UCCA category distributions |
| 11 | **Inference demo** — analyse new sentences with the trained model |
| 12 | Render MTLG dependency graphs (UCCA-coloured, mode-styled edges) |
| 13 | Download trained artefacts (`en_lexicon.json`, `en_trds.json`) |
| 14 | **Launch Streamlit UI** — interactive Train & Inference tabs via public ngrok URL |

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
├── RAG_C4_Training.ipynb          # Colab notebook — clones repo, trains, tests, launches UI
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
