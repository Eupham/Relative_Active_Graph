//! Synonym edge discovery: when two surface forms occupy structurally identical
//! positions across different passages, create or strengthen an edge between them.
//! This is learned synonymy — no wordnet, no hard labels.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId, ModalType};
use crate::arg::{ArgGraph, ArgEdge, EdgeClass};
use crate::types::ModalMode;

/// Track which (TRD, ModalType) slots have been occupied by which NodeIds.
pub struct SlotOccupancyTracker {
    /// (trd_id, mode_bits, category_bits) → list of (node_id, surface, count)
    slot_map: HashMap<(TRDId, u8, u32), Vec<(NodeId, String, usize)>>,
    next_synonym_edge_id: EdgeId,
}

impl SlotOccupancyTracker {
    pub fn new() -> Self {
        Self {
            slot_map: HashMap::new(),
            next_synonym_edge_id: 0x8000_0000,
        }
    }

    /// Record that `node_id` (surface `surface`) occupied slot (trd, mtype) in this passage.
    pub fn observe(&mut self, node_id: NodeId, surface: &str, trd: TRDId, mtype: ModalType) {
        let key = (trd, mtype.mode as u8, mtype.category.0);
        let entry = self.slot_map.entry(key).or_default();
        if let Some(rec) = entry.iter_mut().find(|(nid, _, _)| *nid == node_id) {
            rec.2 += 1;
        } else {
            entry.push((node_id, surface.to_string(), 1));
        }
    }

    /// Return pairs of NodeIds that co-occupy the same slot with count > threshold.
    /// These are synonym candidates.
    pub fn synonym_candidates(&self, min_count: usize) -> Vec<(NodeId, NodeId, f32)> {
        let mut pairs = Vec::new();
        for entries in self.slot_map.values() {
            let qualifying: Vec<_> = entries.iter()
                .filter(|(_, _, c)| *c >= min_count)
                .collect();
            for i in 0..qualifying.len() {
                for j in (i + 1)..qualifying.len() {
                    let strength = (qualifying[i].2.min(qualifying[j].2) as f32)
                        / (qualifying[i].2.max(qualifying[j].2) as f32);
                    pairs.push((qualifying[i].0, qualifying[j].0, strength));
                }
            }
        }
        pairs
    }

    /// Create or strengthen synonym edges in the ARG for all qualifying pairs.
    ///
    /// Creates both forward (src→dst) and reverse (dst→src) edges so that
    /// `synonym_query` traversal works in both directions without a separate pass.
    pub fn materialise_synonym_edges(
        &mut self,
        graph:     &mut ArgGraph,
        min_count: usize,
    ) {
        for (src, dst, strength) in self.synonym_candidates(min_count) {
            let fwd_exists = graph.edge_indices()
                .any(|ei| graph[ei].src == src && graph[ei].dst == dst);
            let rev_exists = graph.edge_indices()
                .any(|ei| graph[ei].src == dst && graph[ei].dst == src);

            let src_idx = graph.node_indices().find(|&i| graph[i].id == src);
            let dst_idx = graph.node_indices().find(|&i| graph[i].id == dst);

            if !fwd_exists {
                if let (Some(si), Some(di)) = (src_idx, dst_idx) {
                    let eid = self.next_synonym_edge_id;
                    self.next_synonym_edge_id += 1;
                    let mut e = ArgEdge::new(eid, src, dst, EdgeClass::DEFAULT, ModalMode::Diamond);
                    e.weight = strength;
                    graph.add_edge(si, di, e);
                }
            } else if let Some(ei) = graph.edge_indices()
                .find(|&i| graph[i].src == src && graph[i].dst == dst)
            {
                // Strengthen existing forward synonym edge.
                graph[ei].weight = (graph[ei].weight + 0.01 * strength).min(1.0);
            }

            // Create or strengthen the reverse edge (dst→src).
            if !rev_exists {
                if let (Some(si), Some(di)) = (src_idx, dst_idx) {
                    let rev_eid = self.next_synonym_edge_id;
                    self.next_synonym_edge_id += 1;
                    let mut e = ArgEdge::new(rev_eid, dst, src, EdgeClass::DEFAULT, ModalMode::Diamond);
                    e.weight = strength;
                    graph.add_edge(di, si, e);
                }
            } else if let Some(ei) = graph.edge_indices()
                .find(|&i| graph[i].src == dst && graph[i].dst == src)
            {
                // Strengthen existing reverse synonym edge.
                graph[ei].weight = (graph[ei].weight + 0.01 * strength).min(1.0);
            }
        }
    }
}

impl Default for SlotOccupancyTracker {
    fn default() -> Self { Self::new() }
}
