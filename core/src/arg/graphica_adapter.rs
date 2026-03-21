//! Graphica-style memoization: canonical key = (structural_hash, atms_env, modal_profile).
//! Two subgraphs with the same topology but different modal compositions are distinct.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use crate::types::{NodeId, EdgeId, Env, CacheKey, ModalMode, TypeCategory};
use crate::arg::{node::ArgNode, edge::ArgEdge};

/// A cached derivation result.
#[derive(Clone, Debug)]
pub struct CachedResult {
    pub key:          CacheKey,
    /// Attribution trace from the prior derivation (reuse avoids redundant computation).
    pub edge_deltas:  HashMap<EdgeId, f32>,
    pub quality:      f32,
}

/// Compute the structural hash of a subgraph (nodes + edges by id, sorted).
pub fn structural_hash(nodes: &[&ArgNode], edges: &[&ArgEdge]) -> u64 {
    let mut h = DefaultHasher::new();
    let mut node_ids: Vec<NodeId> = nodes.iter().map(|n| n.id).collect();
    node_ids.sort_unstable();
    node_ids.hash(&mut h);
    let mut edge_pairs: Vec<(NodeId, NodeId)> = edges.iter().map(|e| (e.src, e.dst)).collect();
    edge_pairs.sort_unstable();
    edge_pairs.hash(&mut h);
    h.finish()
}

/// Build a CacheKey from a subgraph's hash + ATMS environment + dominant modal profile.
pub fn build_key(nodes: &[&ArgNode], edges: &[&ArgEdge], env: Env) -> CacheKey {
    let hash = structural_hash(nodes, edges);
    // Dominant modal mode: mode with most edges.
    let modal_mode = dominant_modal_mode(edges);
    // Dominant category: mode with most nodes.
    let modal_cat  = dominant_type_category(nodes);
    CacheKey { hash, env, modal_mode, modal_cat }
}

fn dominant_modal_mode(edges: &[&ArgEdge]) -> ModalMode {
    let mut counts = [0usize; 3];
    for e in edges {
        match e.modal_mode {
            ModalMode::Diamond => counts[0] += 1,
            ModalMode::Box     => counts[1] += 1,
            ModalMode::Lozenge => counts[2] += 1,
        }
    }
    match counts.iter().enumerate().max_by_key(|&(_, c)| c).map(|(i, _)| i).unwrap_or(0) {
        0 => ModalMode::Diamond,
        1 => ModalMode::Box,
        _ => ModalMode::Lozenge,
    }
}

fn dominant_type_category(nodes: &[&ArgNode]) -> TypeCategory {
    let mut counts = [0usize; 7];
    for n in nodes {
        let idx = match n.mtlg_type.category {
            TypeCategory::Scene       => 0,
            TypeCategory::Process     => 1,
            TypeCategory::State       => 2,
            TypeCategory::Participant => 3,
            TypeCategory::Adverbial   => 4,
            TypeCategory::Connector   => 5,
            TypeCategory::Ground      => 6,
        };
        counts[idx] += 1;
    }
    let max_idx = counts.iter().enumerate().max_by_key(|&(_, c)| c).map(|(i, _)| i).unwrap_or(0);
    [TypeCategory::Scene, TypeCategory::Process, TypeCategory::State,
     TypeCategory::Participant, TypeCategory::Adverbial, TypeCategory::Connector,
     TypeCategory::Ground][max_idx]
}

/// The Graphica memo cache.
pub struct GraphicaCache {
    store: HashMap<CacheKey, CachedResult>,
    hits:  u64,
    misses: u64,
}

impl GraphicaCache {
    pub fn new() -> Self {
        Self { store: HashMap::new(), hits: 0, misses: 0 }
    }

    pub fn get(&mut self, key: &CacheKey) -> Option<&CachedResult> {
        match self.store.get(key) {
            Some(r) => { self.hits += 1; Some(r) }
            None    => { self.misses += 1; None }
        }
    }

    pub fn insert(&mut self, result: CachedResult) {
        self.store.insert(result.key, result);
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { return 0.0; }
        self.hits as f64 / total as f64
    }

    pub fn len(&self) -> usize { self.store.len() }
}

impl Default for GraphicaCache {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalMode, TypeCategory, ModalType};
    use crate::arg::{node::{ArgNode, NodeType}, edge::{ArgEdge, EdgeType}};

    fn make_node(id: NodeId) -> ArgNode {
        ArgNode::new(id, NodeType::Concept, ModalType::default(), (0, 0))
    }

    fn make_edge(id: EdgeId, src: NodeId, dst: NodeId) -> ArgEdge {
        ArgEdge::new(id, src, dst, EdgeType::Composition, ModalMode::Diamond)
    }

    #[test]
    fn same_graph_same_key() {
        let n1 = make_node(1);
        let n2 = make_node(2);
        let e1 = make_edge(10, 1, 2);
        let nodes = vec![&n1, &n2];
        let edges = vec![&e1];
        let key_a = build_key(&nodes, &edges, 0b11);
        let key_b = build_key(&nodes, &edges, 0b11);
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn different_env_different_key() {
        let n1 = make_node(1);
        let nodes = vec![&n1];
        let key_a = build_key(&nodes, &[], 0b01);
        let key_b = build_key(&nodes, &[], 0b10);
        assert_ne!(key_a, key_b);
    }
}
