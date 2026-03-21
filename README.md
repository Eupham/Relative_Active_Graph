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
run inference on new sentences — no GPU required, ~5 min on a free Colab CPU.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eupham/Relative_Active_Graph/blob/claude/colab-c4-training-6sAYV/RAG_C4_Training.ipynb)

### What the notebook covers

| Step | Description |
|------|-------------|
| 1 | Install dependencies (`datasets`, `stanza`, `numpy`, `scipy`, `networkx`, …) |
| 2 | Write the six `lcs/induction/` source modules to the Colab filesystem |
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

---

## Repository Layout

```
Relative_Active_Graph/
├── RAG_C4_Training.ipynb   # Colab training + inference notebook
├── core/                   # Rust reasoning engine (CSRRE)
│   └── src/
│       ├── engine.rs       # 13-step main reasoning loop
│       ├── arg/            # Active Relative Graph (lazy, budget-bounded)
│       ├── atms/           # Base ATMS + BF-ATMS (counterfactuals)
│       ├── adaptive/       # VDBE threshold registry + EMA performance
│       ├── scheduler/      # b/t-level list scheduling + critical path
│       ├── constraints/    # SHACL shapes + sheaf coherence + Z3
│       ├── causal/         # SCM + interventions + counterfactuals
│       ├── semantics/      # MTLG semantics (AMR + DRS + UCCA)
│       ├── generation/     # Progressive deepening + lineariser
│       ├── feedback/       # Attribution + provenance + edge updates
│       ├── rules/          # Rule induction lifecycle (Ruler)
│       └── lcs/            # Language category system
├── cli/                    # Newline-delimited JSON query interface
└── lcs/
    └── induction/          # Python induction pipeline
        ├── mc4_stream.py          # C4 / mC4 streaming (108 languages)
        ├── ud_parser.py           # Stanza UD wrapper
        ├── ud_to_mtlg.py          # UD tree → MTLG modal graph
        ├── morphological_fst.py   # FST decomposition (Turkish, Finnish, …)
        ├── mtlg_inducer.py        # Probabilistic lexicon induction
        ├── trd_bootstrap.py       # TRD k-means crystallisation
        └── hol_to_lc.py           # HOL ↔ λ-calculus conversion
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
