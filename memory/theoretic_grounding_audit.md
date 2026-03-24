# Relative Active Graph — Theoretic Grounding & Bottleneck Audit

Date: 2026-03-24

## Scope

This audit focuses on current implementation behavior vs. stated goals:
- symbolic/structural reasoning with active graph dynamics,
- meaningful training signals (without dense-LLM fallback),
- testable inference and transparent metrics.

## 1) What is currently strong

1. **Explicit symbolic pipeline exists end-to-end.**
   - Python token/graph induction → Rust NDJSON bridge → engine response path is functional.
2. **Clear distinction (in tests/docs) between what is demonstrated vs. aspirational.**
   - Current single-token output and missing persisted lexicon are explicitly acknowledged.
3. **Operational UI loop exists.**
   - Training modal + websocket status + quality/noise traces are usable for rapid iteration.

## 2) Where theory/claims are currently shallow or misaligned

### A. Inference is mostly lexical echo, not robust semantic composition (yet)

- `backend /api/inference/generate` seeds lexicon directly from input words before querying.
- This makes “inference quality” strongly dependent on immediate surface tokens, and weakly tied to longer-horizon learned structure.

**Impact:** perceived reasoning can be overestimated when the system is mostly selecting from freshly registered tokens.

### B. “Graph growth” chart is an estimate, not measured temporal growth

- UI computes per-passage nodes/edges by linearly interpolating final totals over history length.
- This is a placeholder visualization rather than true time-indexed graph evolution.

**Impact:** can mislead training interpretation (e.g., apparent smooth growth that may not exist).

### C. Search labeling overstated formal guarantees

- Deepening implementation uses a practical bounded DFS with inverse-attribution heuristic.
- It was previously labeled as “genuine IDA*”; without admissibility/consistency guarantees, this is too strong.

**Impact:** theoretical framing sounded stronger than implementation evidence.

### D. Knowledge persistence/training continuity is still underpowered

- No robust persisted trained state is loaded by default for inference sessions.
- Cold-start behavior remains likely to regress to single-token outputs.

**Impact:** hard to claim stable “learning” across sessions.

### E. Evaluation lacks theorem-linked metrics

- Mean quality / Poisson / node counts are useful telemetry, but they do not directly test semantic correctness, compositional generalization, or counterfactual validity.

**Impact:** progress can look positive while semantic capability plateaus.

## 3) Main bottlenecks (technical + conceptual)

1. **Bridge granularity bottleneck:** subprocess NDJSON round-trip per interaction increases latency and makes higher-frequency structured updates expensive.
2. **State bottleneck:** missing first-class model state lifecycle (save/load/version/eval) limits reproducibility.
3. **Metric bottleneck:** current dashboard metrics are mostly process indicators, not capability indicators.
4. **Search bottleneck:** heuristic search depth is shallow and heuristic quality is weakly validated.
5. **Category grounding bottleneck:** category assignment remains structurally induced; semantic fidelity to formal targets is not externally validated.

## 4) Corrections (prioritized, no dense-LLM dependency)

## Priority 0 — Truthful instrumentation (immediate)

1. **Rename estimated metrics clearly in UI/API**
   - Keep estimated graph curve, but mark as estimated until true per-step node/edge snapshots are emitted by backend.
2. **Add explicit inference provenance output**
   - Return whether tokens were newly registered in-session vs. retrieved from persisted global state.
3. **Expose session state origin**
   - Add endpoint fields: `state_loaded`, `state_version`, `lexicon_size_loaded`.

## Priority 1 — Capability-grounded evaluation

1. **Create symbolic eval suites**
   - compositional role reversal tests,
   - negation/scope perturbation tests,
   - counterfactual intervention checks.
2. **Track capability metrics**
   - exact semantic target hit rate,
   - type-consistency under perturbation,
   - counterfactual invariance/violation metrics.

## Priority 2 — Persistence and continuity

1. **Implement stable persisted state contract**
   - explicit save checkpoints,
   - deterministic reload path on startup,
   - compatibility/version checks.
2. **Refactor launch defaults**
   - inference should prefer loaded trained state, not only per-request lexical seeding.

## Priority 3 — Search/theory hardening

1. **Formalize heuristic assumptions**
   - document attribution heuristic limits and expected failure modes.
2. **Optional alternative search modes**
   - bounded beam / typed best-first search with explicit objective trade-offs.
3. **Ablation tests**
   - compare search variants under fixed datasets and report stability/quality variance.

## Priority 4 — Throughput and architecture

1. **Move from per-request process overhead toward persistent runtime channels** (same symbolic engine, no dense model requirement).
2. **Batch passage execution where possible** to reduce IPC overhead and improve reproducibility of training timing.

Bottom line: the project has a real symbolic core and runnable stack, but current demonstrations still over-index on pipeline viability and under-index on validated semantic capability. The next gains should come from truthful metrics + persistence + capability-aligned evals.

## Implemented corrections since this audit

- Dataset is explicit in training config/state/UI and pinned to C4 in backend runtime handling.
- Graph growth chart is labeled as an estimate to avoid false precision.
- Bitmask context routing helpers are implemented (`ArgNode` + LCS helpers) for O(1) subset checks.
- Condensed path provenance is retained and can be expanded from canonical shortcut IDs.
- Colab launch no longer aborts outright on repeated React build failure; backend serves fallback UI for continued training/inference access.
