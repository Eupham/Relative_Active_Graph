# Relative Active Graph (RAG)

A **Constraint-Scheduled Reactive Reasoning Engine** implementing multimodal
type-logical semantic composition with adaptive learning.

The system grounds natural language in formal semantics (MTLG / DRS / UCCA / AMR)
and uses situation-relative reasoning with ATMS-backed truth maintenance,
causal SCMs, and variance-adaptive activation thresholds (VDBE).

---

## C4 Training — Colab Notebook

Train the MTLG lexicon and TRD bootstrapper on a streaming subset of the
[C4 corpus](https://huggingface.co/datasets/allenai/c4) (HuggingFace), then
run inference on new sentences and launch the interactive Streamlit UI —
no GPU required, ~5 min on a free Colab CPU.

The notebook **clones this repository directly** so it always runs against the
current codebase — no inline source copies.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eupham/Relative_Active_Graph/blob/master/RAG_C4_Training.ipynb)

### What the notebook covers

| Step | Description |
|------|-------------|
| 1 | Install dependencies (`datasets`, `stanza`, `numpy`, `scipy`, `networkx`, `streamlit`, …) |
| 2 | **Clone repo from GitHub** — source modules loaded from the live codebase |
| 3 | Configure demo parameters (`TRAIN_SAMPLES`, `HELD_OUT`, `N_TRD_CLUSTERS`) |
| 4 | Download Stanza UD model for the target language |
| 5 | **Stream C4** via HuggingFace streaming API (no full download) |
| 6 | Parse sentences → Universal Dependencies → MTLG modal graphs (◇ / □ / ◊) |
| 7 | **Induce MTLG lexicon** — MLE over `(lemma, mode, ucca_cat, arity)` |
| 8 | **Bootstrap TRD clusters** — k-means over modal type profile vectors |
| 9 | Evaluate parse accuracy + TRD coverage on held-out sentences |
| 10 | Visualise modal mode and UCCA category distributions |
| 11 | **Inference demo** — analyse new sentences with the trained model |
| 12 | Render MTLG dependency graphs with NetworkX (UCCA-coloured, mode-styled) |
| 13 | Download trained artefacts (`en_lexicon.json`, `en_trds.json`) |
| 14 | **Launch Streamlit UI** — interactive training & inference via public ngrok URL |

---

## Streamlit UI

Run the interactive web interface locally:

```bash
pip install -r lcs/requirements.txt
streamlit run app.py
```

The UI provides two tabs:
- **Train** — stream C4, induce lexicon, bootstrap TRDs with live progress
- **Inference & Visualization** — analyse sentences and render MTLG graphs

---

## Repository Layout

```
Relative_Active_Graph/
├── RAG_C4_Training.ipynb          # Colab training + inference notebook (clones repo)
├── app.py                         # Streamlit web UI (train + inference)
├── test_e2e_english.py            # End-to-end English pipeline test
├── test_summary.md                # Test execution results
├── core/                          # Rust reasoning engine (CSRRE)
│   └── src/
│       ├── engine.rs              # 13-step main reasoning loop
│       ├── arg/                   # Active Relative Graph (lazy, budget-bounded)
│       ├── atms/                  # Base ATMS + BF-ATMS (counterfactuals)
│       ├── adaptive/              # VDBE threshold registry + EMA performance
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
    │   ├── ud_parser.py           # Stanza UD wrapper
    │   ├── ud_to_mtlg.py          # UD tree → MTLG modal graph
    │   ├── morphological_fst.py   # FST decomposition (Turkish, Finnish, …)
    │   ├── mtlg_inducer.py        # Probabilistic lexicon induction (MLE)
    │   ├── trd_bootstrap.py       # TRD k-means crystallisation
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

## End-to-End Tests

```bash
# Python pipeline integration test
python lcs/induction/test_pipeline.py

# Full English end-to-end (requires Rust binary)
cargo build --release
python test_e2e_english.py
```
