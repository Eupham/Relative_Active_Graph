//! Linguistic Conversion System: character sequence -> MTLG modal graph.
//!
//! - `ud_types`       — CharToken / CharSequence + hash utilities
//! - `converter`      — Character-to-MTLG converter (Unicode properties only)
//! - `category_registry` — Runtime TypeCategory registry
//! - `bootstrap_loader`  — Loads Phase 0 bootstrap artefacts from JSON

pub mod ud_types;
pub mod converter;
pub mod category_registry;
pub mod bootstrap_loader;

pub use ud_types::{CharToken, CharSequence, deprel_hash};
pub use converter::{
    CharTokenStructure,
    MtlgNode, MtlgEdge, MtlgGraph,
    CategoryInducer,
    char_sequence_to_mtlg,
    resolve_deferred,
    structure_to_vector,
};
pub use category_registry::{CategoryRegistry, CategoryEntry};
pub use bootstrap_loader::{BootstrapArtefacts, TrdProfile, load_bootstrap};
