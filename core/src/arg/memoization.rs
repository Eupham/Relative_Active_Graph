//! Graphica-style memoization: canonical key = (structural_hash, atms_env, modal_profile).
//! Two subgraphs with the same topology but different modal compositions are distinct.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use crate::types::{NodeId, EdgeId, Env, CacheKey, ModalMode, TypeCategory};
use crate::arg::{node::ArgNode, edge::ArgEdge};

pub const MIN_SHORTCUT_TRAVERSALS: u32 = 10;

#[derive(Clone, Debug)]
pub struct CachedResult {
    pub key:                   CacheKey,
    pub edge_deltas:           HashMap<EdgeId, f32>,
    pub quality:               f32,
    pub traversal_count:       u32,
    pub shortcut_canonical_id: Option<u64>,
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
    let modal_mode = dominant_modal_mode(edges);
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
    let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for n in nodes { *counts.entry(n.mtlg_type.category.id()).or_insert(0) += 1; }
    counts.into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(id, _)| TypeCategory(id))
        .unwrap_or(TypeCategory::DEFAULT)
}

/// The Graphica memo cache.
pub struct GraphicaCache {
    store: HashMap<CacheKey, CachedResult>,
    condensed_paths: HashMap<u64, Vec<EdgeId>>,
    hits:  u64,
    misses: u64,
}

impl GraphicaCache {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
            condensed_paths: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, key: &CacheKey) -> Option<&CachedResult> {
        match self.store.get(key) {
            Some(r) => { self.hits += 1; Some(r) }
            None    => { self.misses += 1; None }
        }
    }

    pub fn record_traversal(&mut self, key: &CacheKey) -> u32 {
        if let Some(r) = self.store.get_mut(key) { r.traversal_count += 1; r.traversal_count }
        else { 0 }
    }

    pub fn insert(&mut self, result: CachedResult) { self.store.insert(result.key, result); }

    /// Emit a shortcut edge once MIN_SHORTCUT_TRAVERSALS is reached.
    /// Returns Some on the first eligible call, None thereafter (no duplicates).
    pub fn try_emit_shortcut(
        &mut self,
        key:            &CacheKey,
        src:            NodeId,
        dst:            NodeId,
        canonical_mode: crate::types::ModalMode,
        path_weights:   &[f32],
        path_edge_ids:  &[EdgeId],
        next_edge_id:   EdgeId,
    ) -> Option<crate::arg::edge::ArgEdge> {
        use crate::arg::edge::{ArgEdge, EdgeClass};
        use crate::arg::math_utils::normalize_path_weight;
        let result = self.store.get_mut(key)?;
        if result.traversal_count < MIN_SHORTCUT_TRAVERSALS { return None; }
        if result.shortcut_canonical_id.is_some() { return None; }
        let canonical_id = key.hash ^ (canonical_mode as u64).wrapping_mul(0x9e3779b97f4a7c15);
        result.shortcut_canonical_id = Some(canonical_id);
        self.condensed_paths.insert(canonical_id, path_edge_ids.to_vec());
        let weight = normalize_path_weight(path_weights).clamp(0.01, 1.0);
        let mut edge = ArgEdge::new(next_edge_id, src, dst, EdgeClass::DEFAULT, canonical_mode);
        edge.weight       = weight;
        edge.canonical_id = Some(canonical_id);
        Some(edge)
    }

    /// Expand a condensed shortcut to its original path edge IDs.
    pub fn expand_shortcut(&self, canonical_id: u64) -> Option<&[EdgeId]> {
        self.condensed_paths.get(&canonical_id).map(Vec::as_slice)
    }

    pub fn update_quality(&mut self, key: &CacheKey, q: f32) {
        if let Some(r) = self.store.get_mut(key) {
            r.quality = 0.9 * r.quality + 0.1 * q;
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
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
    use crate::arg::{node::{ArgNode, NodeClass}, edge::{ArgEdge, EdgeClass}};

    fn make_node(id: NodeId) -> ArgNode {
        ArgNode::new(id, NodeClass::DEFAULT, ModalType::default(), (0, 0))
    }

    fn make_edge(id: EdgeId, src: NodeId, dst: NodeId) -> ArgEdge {
        ArgEdge::new(id, src, dst, EdgeClass::DEFAULT, ModalMode::Diamond)
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

    #[test]
    fn shortcut_emitted_after_threshold() {
        use crate::types::ModalMode;
        let n1 = make_node(1); let n2 = make_node(2); let e1 = make_edge(10, 1, 2);
        let key = build_key(&[&n1, &n2], &[&e1], 0b11);
        let mut cache = GraphicaCache::new();
        cache.insert(CachedResult {
            key, edge_deltas: Default::default(), quality: 0.8,
            traversal_count: 0, shortcut_canonical_id: None,
        });
        for _ in 0..9 {
            cache.record_traversal(&key);
            assert!(cache.try_emit_shortcut(&key, 1, 2, ModalMode::Diamond, &[0.8], &[10], 99).is_none());
        }
        cache.record_traversal(&key);
        let sc = cache.try_emit_shortcut(&key, 1, 2, ModalMode::Diamond, &[0.8], &[10, 11], 99);
        assert!(sc.is_some() && sc.unwrap().canonical_id.is_some());
        let canonical_id = cache.store.get(&key).and_then(|r| r.shortcut_canonical_id).unwrap();
        assert_eq!(cache.expand_shortcut(canonical_id), Some(&[10, 11][..]));
        assert!(cache.try_emit_shortcut(&key, 1, 2, ModalMode::Diamond, &[0.8], &[10], 100).is_none());
    }
}
