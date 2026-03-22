//! PassageContext: a multi-sentence training unit.
//! Keeps the ARG context open across sentence boundaries.
//! Attribution is accumulated per sentence but only flushed
//! (and backward-propagated) when the passage closes.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId};
use crate::arg::{ArgSearch, ArgNode, ArgEdge, EdgeClass};
use crate::types::{Env, ModalMode};

/// FNV-1a over the ordered pair (src, dst) with a class-marker XOR.
/// Keeps sequential edge IDs in a distinct space from structural edges.
pub fn sequential_edge_id(src: NodeId, dst: NodeId) -> EdgeId {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME:  u64 = 0x0000_0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for &b in &src.to_le_bytes() { h = h.wrapping_mul(FNV_PRIME) ^ b as u64; }
    for &b in &dst.to_le_bytes() { h = h.wrapping_mul(FNV_PRIME) ^ b as u64; }
    h ^ 0x6000_0000_0000_0000u64
}

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

    /// Add one token to the accumulated graph and return the `ArgSearch`
    /// representing the state *before* this token was added.
    ///
    /// The returned search is used to compute `VocabDistribution` for this step:
    /// the model predicts the current token from all prior context.
    /// Only after prediction is this token committed to the context graph.
    ///
    /// Also creates a SEQUENTIAL edge from `prev_node_id` to
    /// `expected_node_id`, establishing bigram position signal.
    pub fn step(
        &mut self,
        expected_node_id: NodeId,
        node_pool:        &[ArgNode],
        edge_pool:        &[ArgEdge],
        active_env:       Env,
        theta_alpha:      f64,
        theta_rho:        f64,
        prev_node_id:     Option<NodeId>,
    ) -> ArgSearch {
        // Snapshot current context for prediction (before adding this token).
        let search = self.build_graph(active_env, theta_alpha, theta_rho);

        // Commit this token's nodes and edges to the accumulated context.
        for n in node_pool {
            self.node_map.entry(n.id).or_insert_with(|| n.clone());
        }
        for e in edge_pool {
            self.edge_map.entry(e.id).or_insert_with(|| e.clone());
        }

        // Sequential edge from previous token to this token.
        if let Some(prev) = prev_node_id {
            let seq_id = sequential_edge_id(prev, expected_node_id);
            let seq_edge = ArgEdge::new(
                seq_id, prev, expected_node_id,
                EdgeClass::SEQUENTIAL, ModalMode::Diamond,
            );
            self.edge_map.entry(seq_id).or_insert(seq_edge);
        }

        self.sentence_count += 1;
        search
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

    /// Perform one inference step: predict the next token from current context,
    /// then commit the prediction to the accumulated graph.
    ///
    /// Returns:
    /// - The predicted NodeId (argmax of VocabDistribution).
    /// - The probability of that prediction (for stopping condition).
    ///
    /// Mirrors `step()` exactly: build graph from current context, score it,
    /// select best node, commit it, create sequential edge from prev.
    /// No attribution is applied.
    pub fn decode_step(
        &mut self,
        node_pool:    &[ArgNode],
        edge_pool:    &[ArgEdge],
        active_env:   Env,
        theta_alpha:  f64,
        theta_rho:    f64,
        prev_node_id: Option<NodeId>,
    ) -> Option<(NodeId, f32)> {
        use crate::generation::VocabDistribution;

        // Snapshot context before committing candidates.
        let search = self.build_graph(active_env, theta_alpha, theta_rho);
        let dist = VocabDistribution::from_graph(&search.graph, active_env);

        // Select highest-probability node.
        let (predicted_id, predicted_prob) = dist.probs.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .copied()?;

        // Commit the candidate pool so future steps see these nodes.
        for n in node_pool {
            self.node_map.entry(n.id).or_insert_with(|| n.clone());
        }
        for e in edge_pool {
            self.edge_map.entry(e.id).or_insert_with(|| e.clone());
        }

        // Sequential edge: same logic as training.
        if let Some(prev) = prev_node_id {
            let seq_id = sequential_edge_id(prev, predicted_id);
            let seq_edge = ArgEdge::new(
                seq_id, prev, predicted_id,
                EdgeClass::SEQUENTIAL, ModalMode::Diamond,
            );
            self.edge_map.entry(seq_id).or_insert(seq_edge);
        }

        self.sentence_count += 1;
        Some((predicted_id, predicted_prob))
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
