//! Linguistic Conversion System: tokenised text → MTLG modal graph.
//! No external parser. Categories assigned by bisimulation partition refinement (Paige & Tarjan 1987).

pub mod token_types;
pub mod converter;
pub mod category_registry;
pub mod bootstrap_loader;

pub use token_types::{Token, TokenSentence, TokenStructure, fnv_hash, fnv1a_64_bytes, stable_node_id, extract_features};
pub use converter::{
    MtlgNode, MtlgEdge, MtlgGraph,
    CategoryInducer,
    sentence_to_mtlg,
    resolve_deferred,
    structure_to_vector,
};
pub use category_registry::{CategoryRegistry, CategoryEntry};
pub use bootstrap_loader::{BootstrapArtefacts, TrdProfile, load_bootstrap};
