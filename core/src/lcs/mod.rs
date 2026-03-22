//! Linguistic Conversion System: UD dependency tree → MTLG modal graph.
//!
//! - `ud_types`          — UD CoNLL-U token and tree types + structural deprel predicates
//! - `converter`         — UD-to-MTLG converter (structural evidence, no UPOS comparison)
//! - `category_registry` — Runtime TypeCategory registry (cluster IDs + centroids)
//! - `bootstrap_loader`  — Loads Phase 0 bootstrap artefacts from JSON

pub mod ud_types;
pub mod converter;
pub mod category_registry;
pub mod bootstrap_loader;

pub use ud_types::{UdToken, UdTree};
pub use converter::{
    MtlgNode, MtlgEdge, MtlgGraph,
    CategoryInducer,
    ud_tree_to_mtlg,
    resolve_deferred,
};
pub use category_registry::{CategoryRegistry, CategoryEntry};
pub use bootstrap_loader::{BootstrapArtefacts, TrdProfile, load_bootstrap};
