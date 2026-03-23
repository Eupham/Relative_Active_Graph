//! Linguistic Conversion System: tokenised text → MTLG modal graph.
//! No external parser. Categories learned online via k-means.

pub mod token_types;
pub mod converter;
pub mod category_registry;
pub mod bootstrap_loader;

pub use token_types::{Token, TokenSentence, TokenStructure, fnv_hash, stable_node_id, extract_features};
pub use converter::{
    MtlgNode, MtlgEdge, MtlgGraph,
    CategoryInducer,
    sentence_to_mtlg,
    resolve_deferred,
    structure_to_vector,
};
pub use category_registry::{CategoryRegistry, CategoryEntry};
pub use bootstrap_loader::{BootstrapArtefacts, TrdProfile, load_bootstrap};
