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
/// Delegates core hashing to token_types::fnv1a_64_bytes.
pub fn sequential_edge_id(src: NodeId, dst: NodeId) -> EdgeId {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&src.to_le_bytes());
    bytes[8..].copy_from_slice(&dst.to_le_bytes());
    crate::lcs::fnv1a_64_bytes(&bytes) ^ 0x6000_0000_0000_0000u64
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

    /// Perform one inference step using L-System expansion to generate candidates,
    /// then commit the prediction to the accumulated graph.
    ///
    /// Returns:
    /// - The predicted NodeId (first non-empty token from LSystemExpander, or
    ///   argmax of VocabDistribution when no rules apply — identity expansion).
    /// - The probability of that prediction (for stopping condition).
    ///
    /// The axiom is the previous node (or highest-attribution node if no prev).
    /// LSystemExpander::expand drives generative decoding via MetaGrammarEngine rules.
    /// When the rule set is empty the L-System is the identity: every node expands to
    /// itself, which degrades gracefully to the VocabDistribution argmax baseline.
    pub fn decode_step(
        &mut self,
        node_pool:    &[ArgNode],
        edge_pool:    &[ArgEdge],
        active_env:   Env,
        theta_alpha:  f64,
        theta_rho:    f64,
        prev_node_id: Option<NodeId>,
        meta_grammar: &crate::semantics::MetaGrammarEngine,
    ) -> Option<(NodeId, f32)> {
        use crate::generation::{LSystemExpander, GeneratedToken, VocabDistribution};

        // Snapshot context before committing candidates.
        let search = self.build_graph(active_env, theta_alpha, theta_rho);

        // Choose axiom: the previous node if present, else highest-attribution node.
        let axiom_id: Option<NodeId> = prev_node_id.or_else(|| {
            self.node_map.values()
                .filter(|n| n.atms_label == 0 || (n.atms_label & active_env) != 0)
                .max_by(|a, b| a.attribution_score.partial_cmp(&b.attribution_score)
                    .unwrap_or(std::cmp::Ordering::Equal))
                .map(|n| n.id)
        });

        // Run L-System expansion if we have an axiom node.
        let lsystem_prediction: Option<NodeId> = axiom_id
            .and_then(|aid| self.node_map.get(&aid).cloned())
            .and_then(|axiom_node| {
                let expander = LSystemExpander::new(meta_grammar);
                let tokens = expander.expand(&axiom_node, active_env, 0);
                // First token that resolves to a known NodeId wins.
                tokens.into_iter().find_map(|t| match t {
                    GeneratedToken::Node(nid) if self.node_map.contains_key(&nid) => Some(nid),
                    GeneratedToken::Leaf(_) => axiom_id, // surface leaf → reuse axiom id
                    _ => None,
                })
            });

        // Fall back to VocabDistribution argmax (identity-expansion path).
        let dist = VocabDistribution::from_graph(&search.graph, active_env);
        let (predicted_id, predicted_prob) = lsystem_prediction
            .and_then(|nid| {
                // Use the VocabDistribution probability for the L-System prediction if available.
                dist.probs.iter().find(|&&(id, _)| id == nid).copied()
            })
            .or_else(|| {
                dist.probs.iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .copied()
            })?;

        // Commit the candidate pool so future steps see these nodes.
        for n in node_pool {
            self.node_map.entry(n.id).or_insert_with(|| n.clone());
        }
        for e in edge_pool {
            self.edge_map.entry(e.id).or_insert_with(|| e.clone());
        }

        // Suppress the selected node so the next decode step picks a different one.
        if let Some(node) = self.node_map.get_mut(&predicted_id) {
            node.attribution_score = f32::NEG_INFINITY;
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
