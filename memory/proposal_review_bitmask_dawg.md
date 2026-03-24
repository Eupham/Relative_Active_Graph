# Proposal Review: ATMS Bitmask + Condensed Path Equivalence

Date: 2026-03-24

## What I reviewed first (existing stack)

- `csrre-core` already depends on `egg` (e-graph), `graphica`, `numerica`, and uses `Env = u64` as an ATMS bitmask.
- `ArgNode` already carries `atms_label: Env`.
- `lcs::token_types::contextual_node_id` already mixes structure + env + mode into a deterministic 64-bit ID.
- `GraphicaCache` already emits shortcut edges with a `canonical_id`.

## Proposal fit check

### 1) Tuple identity `(Base_NodeId, Context_Bitmask)` and O(1) context search

This is compatible with current architecture.

Implemented:
- Added explicit base/context helpers in `lcs::token_types`:
  - `base_node_id_from_structure`
  - `contextual_node_id_from_base`
  - `context_mask_matches`
- Added `ArgNode::matches_context_mask` for direct bitwise routing.

This keeps deterministic symbolic behavior and avoids string-scanning during context filtering.

### 2) Condensed path with reversible provenance (DAWG-like idea)

Full DAWG integration is larger than this patch, but current e-graph + memo infrastructure supports a practical first step:

Implemented:
- `GraphicaCache` now stores `canonical_id -> constituent edge IDs` for emitted shortcuts.
- Added `expand_shortcut(canonical_id)` to recover the original path edge IDs.
- Engine integration now opportunistically emits a shortcut edge from high-weight traversals, so condensed-path provenance is not dead code.

This preserves explainability/provenance for condensed traversal paths while keeping fast shortcut traversal.

## What was intentionally not overclaimed

- No claim that this is a full DAWG implementation.
- No claim that canonical shortcut expansion is complete proof reconstruction; it is a concrete provenance hook for path constituents.

## Concrete follow-through added in this patch

1. **Bitmask query primitive is now operational** in ARG node logic and available in LCS helpers.
2. **Condensed-path provenance is now exercised by engine execution path**, not only tests.
3. **Colab notebook flow now hard-fails less often**: after repeated npm build failures, the launcher continues and backend serves a fallback UI so training/inference can still be run.
