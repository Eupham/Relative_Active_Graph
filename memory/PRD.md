# PRD: Relative Active Graph (CSRRE) Bug Fixes

## Original Problem Statement
Fix 6 identified code issues in the Relative_Active_Graph repository (Rust/Python codebase for a CSRRE engine).

## Architecture
- **Core Engine**: Rust workspace (core/ + cli/) with ATMS, ARG, LCS subsystems
- **Training Pipeline**: Python (lcs/training/) with subprocess bridge to Rust CLI
- **Induction**: Python (lcs/induction/) tokenizer and graph tools

## What's Been Implemented (Jan 2026)

### Bug Fixes Applied
1. **ContextStack::shift() ATMS bit leak** (context_stack.rs) — Added `try_reclaim_pending(&[])` between pop/push to recycle freed bits
2. **normalize_path_weight LSE→geometric mean regression** (numerica_adapter.rs) — Restored numerically stable Log-Sum-Exp implementation
3. **register_lexicon fire-and-forget** (rust_bridge.py) — Changed to use `self.send()` to consume potential engine responses
4. **tokenizer.py 32-bit FNV mismatch** (tokenizer.py) — Updated `_stable_node_id` to 64-bit FNV-1a matching all other files

### Issues Confirmed Already Correct
5. FNV-1 to FNV-1a change — consistently applied, no persisted state concern
6. sentence_to_mtlg inducer fix — genuine fix already in place

## Testing Status
- All 132 Rust tests pass (cargo test)
- Python hash consistency verified across 3 files
- All ATMS tests (27) pass including sequential_1000_contexts_no_exhaustion

## Backlog / Future
- P1: Add integration test for 64+ sequential shift() calls specifically
- P2: Add Python unit tests for tokenizer._stable_node_id
- P2: Consider state migration tooling if save/load is used with old FNV-1 hashes
