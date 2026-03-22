//! VocabDistribution: softmax over active ARG nodes for P(token|context).
//! Maps cross-entropy gradient to Quality signal for teacher forcing.

use crate::types::{NodeId, EdgeId, Quality, Env};
use crate::arg::ArgGraph;

/// Probability distribution over active nodes, derived from attribution scores.
pub struct VocabDistribution {
    /// (node_id, edge_id) → probability under softmax.
    pub probs: Vec<(NodeId, EdgeId, f32)>,
}

impl VocabDistribution {
    /// Build a softmax distribution over the active nodes in the ARG.
    ///
    /// Each node contributes its `attribution_score` as the logit.
    /// Edge IDs are taken from the first outgoing edge of each node (or 0 if none).
    pub fn from_graph(graph: &ArgGraph, active_env: Env) -> Self {
        use petgraph::visit::EdgeRef;

        let active: Vec<(NodeId, EdgeId, f32)> = graph.node_indices()
            .filter(|&ni| {
                let n = &graph[ni];
                (n.atms_label & active_env) != 0
            })
            .map(|ni| {
                let n = &graph[ni];
                let node_id = n.id;
                let score   = n.attribution_score;
                // Take the first outgoing edge ID as the "edge for this token".
                let edge_id = graph.edges(ni)
                    .next()
                    .map(|e| graph[e.id()].id)
                    .unwrap_or(0);
                (node_id, edge_id, score)
            })
            .collect();

        if active.is_empty() {
            return Self { probs: vec![] };
        }

        // Softmax over attribution scores.
        let max_score = active.iter().map(|&(_, _, s)| s).fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = active.iter().map(|&(_, _, s)| (s - max_score).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>() + 1e-12;
        for e in &mut exps { *e /= sum; }

        let probs = active.into_iter()
            .zip(exps)
            .map(|((nid, eid, _), p)| (nid, eid, p))
            .collect();

        Self { probs }
    }

    /// Probability of the node associated with `edge_id`.
    pub fn probability_of_edge(&self, edge_id: EdgeId) -> f32 {
        self.probs.iter()
            .find(|&&(_, eid, _)| eid == edge_id)
            .map(|&(_, _, p)| p)
            .unwrap_or(0.0)
    }

    /// Compute a Quality signal via cross-entropy.
    ///
    /// CE = −log(P(expected)), clamped. Mapped to Quality via `Quality::from_ce`.
    /// `was_correct`: whether the predicted token matched the expected token.
    pub fn ce_quality(&self, expected_edge_id: EdgeId, was_correct: bool) -> Quality {
        let p = self.probability_of_edge(expected_edge_id).max(1e-12);
        Quality::from_ce(p, was_correct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgGraph, ArgNode, NodeClass};
    use crate::types::{ModalType, ModalMode, TypeCategory, Direction, Quality};
    use petgraph::stable_graph::StableGraph;

    #[test]
    fn softmax_sums_to_one() {
        let mut g: ArgGraph = StableGraph::new();
        for id in 1u64..=4 {
            let mt = ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 0, Direction::Right);
            let mut n = ArgNode::new(id, NodeClass::DEFAULT, mt, (0, 0));
            n.attribution_score = id as f32 * 0.25;
            n.atms_label = 0b1;
            g.add_node(n);
        }
        let dist = VocabDistribution::from_graph(&g, 0b1);
        let total: f32 = dist.probs.iter().map(|&(_, _, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-5, "probs should sum to 1, got {}", total);
    }

    #[test]
    fn ce_quality_low_prob_gives_high_update_signal() {
        // from_ce returns (1 - p_predicted) when correct: low prob → large update signal.
        let mut g: ArgGraph = StableGraph::new();
        let mt = ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right);
        let mut n1 = ArgNode::new(1, NodeClass::DEFAULT, mt, (0, 0));
        n1.attribution_score = 0.1; // low prob (model uncertain)
        n1.atms_label = 0b1;
        let mut n2 = ArgNode::new(2, NodeClass::DEFAULT, mt, (0, 0));
        n2.attribution_score = 10.0; // dominant (wrong answer)
        n2.atms_label = 0b1;
        g.add_node(n1);
        g.add_node(n2);
        let dist = VocabDistribution::from_graph(&g, 0b1);
        // n1 has low probability; when it is the correct answer, update signal = 1 - p ≈ high
        let p_n1 = dist.probs.iter().find(|&&(nid, _, _)| nid == 1).map(|&(_, _, p)| p).unwrap_or(0.0);
        let q = Quality::from_ce(p_n1, true);
        assert!(q.as_f32() > 0.5,
            "low-prob correct prediction gives large CE update signal (1 - p ≈ high), got {}", q.as_f32());
    }
}
