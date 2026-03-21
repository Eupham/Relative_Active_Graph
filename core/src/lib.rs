//! CSRRE Core: Constraint-Scheduled Reactive Reasoning Engine.
//!
//! Architecture:
//! - `atms`        — Horn-clause ATMS (base) + BF-ATMS (causal counterfactual)
//! - `arg`         — Active Relative Graph: context stack, TRs, lifting, e-graph, graphica
//! - `adaptive`    — TRD-relative VDBE thresholds + per-TRD performance EMA
//! - `scheduler`   — b/t-level list scheduling, ATMS-integrated state machine
//! - `constraints` — SHACL structural validation, sheaf coherence, modal type checking
//! - `causal`      — SCM, causal bootstrapping, do(e=absent) interventions
//! - `semantics`   — MTLG semantics: sentence_level (AMR-like) + discourse_level (DRS)
//! - `rules`       — Ruler-style rule induction, lifecycle, TRD relevance gate
//! - `feedback`    — Attribution (Δ), edge weight update, provenance audit
//! - `generation`  — Hypothesis ranking, lazy traversal, progressive deepening, linearization
//! - `engine`      — Orchestrating execution pipeline (Section 7 of CSRRE spec)

pub mod adaptive;
pub mod arg;
pub mod atms;
pub mod causal;
pub mod constraints;
pub mod engine;
pub mod feedback;
pub mod generation;
pub mod rules;
pub mod scheduler;
pub mod semantics;
pub mod types;

pub use engine::{Engine, Query, QueryResult};
pub use types::{ModalMode, ModalType, TypeCategory, Quality, Env, NodeId, EdgeId, ContextId, TRDId};
