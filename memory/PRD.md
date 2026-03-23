# PRD: Relative Active Graph (CSRRE) — Numerica/Graphica Consolidation

## Original Problem Statement
1. Fix 6 identified code bugs (ATMS bit leak, LSE→geometric mean, register_lexicon, FNV hash, converter, tokenizer)
2. Remove all symbolica references — use Rust from numerica_adapter and graphica_adapter as the canonical libraries
3. Eliminate manual inline reimplementations — abstract math to numerica, hashing to token_types/graphica

## Architecture
- **Core Engine**: Rust workspace (core/ + cli/) with ATMS, ARG, LCS, generation, semantics subsystems
- **Training Pipeline**: Python (lcs/training/) subprocess bridge to Rust CLI
- **Induction**: Python (lcs/induction/) tokenizer and graph tools

## What's Been Implemented (Jan 2026)

### Phase 1: Bug Fixes
1. **ContextStack::shift() ATMS bit leak** — Added `try_reclaim_pending(&[])` between pop/push
2. **normalize_path_weight** — Restored LSE from geometric mean
3. **register_lexicon** — Changed fire-and-forget to `self.send()` to consume responses
4. **tokenizer.py** — Updated `_stable_node_id` to 64-bit FNV-1a

### Phase 2: Numerica/Graphica Consolidation
5. **Removed all symbolica references** — lifting.rs comment, numerica_adapter.rs module doc
6. **numerica_adapter now owns**:
   - `normalize_path_weight` (LSE)
   - `softmax` (extracted from vocab_distribution.rs)
   - `compute_depth` (extracted from node.rs)
   - `sample_discrete_noise` (Poisson)
7. **token_types now owns all FNV hashing**:
   - Added `fnv1a_64_bytes(&[u8]) -> u64` as the canonical 64-bit byte-level FNV-1a
   - `stable_node_id` now delegates to `fnv1a_64_bytes`
   - Exported via `lcs::fnv1a_64_bytes`
8. **Eliminated 7 inline FNV reimplementations**:
   - vocab_distribution.rs → `crate::lcs::stable_node_id`
   - linearizer.rs → `crate::lcs::stable_node_id`
   - converter.rs → `fnv1a_64_bytes`
   - passage_context.rs → `crate::lcs::fnv1a_64_bytes`
   - meta_grammar.rs → `crate::lcs::fnv1a_64_bytes`
   - mtlg_semantics.rs → `crate::lcs::fnv1a_64_bytes`
   - bootstrap_loader.rs → `crate::lcs::fnv1a_64_bytes`
9. **Fixed 4 FNV-1 (wrong order) bugs** — passage_context, bootstrap_loader, meta_grammar, mtlg_semantics all silently had multiply-then-xor instead of xor-then-multiply

### Testing
- 132/132 Rust tests pass (128 unit + 4 integration)
- Python hash consistency verified across tokenizer, c4_sequence_extractor, sequential_trainer
- Zero remaining inline FNV constants outside token_types.rs
- Zero remaining symbolica references
- Zero FNV-1 (wrong order) instances

## Backlog
- P1: Add numerica unit tests for `softmax` and `compute_depth`
- P2: Consider extracting `clamp_weight` to numerica if edge weight clamping patterns grow
- P2: Add Python unit tests for `tokenizer._stable_node_id` matching Rust output
- P3: State migration tooling if save/load is used with old FNV-1 hashes
