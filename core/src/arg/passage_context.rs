//! PassageContext: a multi-sentence training unit.
//! Keeps the ARG context open across sentence boundaries.
//! Attribution is accumulated per sentence but only flushed
//! (and backward-propagated) when the passage closes.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId};
use crate::arg::{ArgSearch, ArgNode, ArgEdge};
use crate::types::Env;

/// Accumulated CE signal across one passage.
pub struct PassageSignal {
    /// edge_id → cumulative signed CE quality
    pub edge_signals: HashMap<EdgeId, f32>,
    pub token_count:  usize,
    pub total_ce:     f32,
}

impl PassageSignal {
    pub fn new() -> Self {
        Self { edge_signals: HashMap::new(), token_count: 0, total_ce: 0.0 }
    }

    pub fn accumulate(&mut self, edge_id: EdgeId, signal: f32) {
        *self.edge_signals.entry(edge_id).or_insert(0.0) += signal;
        self.total_ce += signal.abs();
        self.token_count += 1;
    }

    /// Mean CE over the passage.
    pub fn mean_ce(&self) -> f32 {
        if self.token_count == 0 { return 0.0; }
        self.total_ce / self.token_count as f32
    }
}

impl Default for PassageSignal {
    fn default() -> Self { Self::new() }
}

/// Maintains a merged ARG across all sentences in a passage.
pub struct PassageContext {
    pub trd:            TRDId,
    pub signal:         PassageSignal,
    /// Merged node pool: all nodes seen in this passage, keyed by NodeId.
    pub node_map:       HashMap<NodeId, ArgNode>,
    /// All edges seen in this passage, keyed by EdgeId.
    pub edge_map:       HashMap<EdgeId, ArgEdge>,
    pub sentence_count: usize,
}

impl PassageContext {
    pub fn new(trd: TRDId) -> Self {
        Self {
            trd,
            signal:         PassageSignal::new(),
            node_map:       HashMap::new(),
            edge_map:       HashMap::new(),
            sentence_count: 0,
        }
    }

    /// Add all nodes and edges from a sentence's pools.
    /// Nodes are deduplicated by NodeId; first occurrence wins.
    pub fn absorb_sentence(&mut self, nodes: &[ArgNode], edges: &[ArgEdge]) {
        for n in nodes {
            self.node_map.entry(n.id).or_insert_with(|| n.clone());
        }
        for e in edges {
            self.edge_map.entry(e.id).or_insert_with(|| e.clone());
        }
        self.sentence_count += 1;
    }

    /// Build the merged ARG from all absorbed nodes/edges.
    pub fn build_graph(&self, active_env: Env, theta_alpha: f64, theta_rho: f64) -> ArgSearch {
        let mut search = ArgSearch::new(active_env, theta_alpha, theta_rho);
        for n in self.node_map.values() {
            search.try_activate(n.clone());
        }
        for e in self.edge_map.values() {
            search.try_add_edge(e.clone());
        }
        search
    }
}
