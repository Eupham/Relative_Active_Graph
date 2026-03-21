//! Linguistic Conversion System: UD dependency tree → MTLG modal graph.
//!
//! - `ud_types`  — UD CoNLL-U token and tree types + structural deprel predicates
//! - `converter` — UD-to-MTLG converter (structural evidence, no UPOS comparison)

pub mod ud_types;
pub mod converter;

pub use ud_types::{UdToken, UdTree};
pub use converter::{
    MtlgNode, MtlgEdge, MtlgGraph,
    CategoryInducer,
    ud_tree_to_mtlg,
    resolve_deferred,
};
