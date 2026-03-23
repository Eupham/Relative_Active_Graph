//! VocabDistribution: softmax over active ARG nodes for P(token|context).
//! Maps cross-entropy gradient to Quality signal for teacher forcing.
//! Vocabulary identity is the NodeId itself — not a transient edge ID.

use crate::types::{NodeId, Quality, Env};
use crate::arg::ArgGraph;

/// Probability distribution over active nodes, derived from attribution scores.
pub struct VocabDistribution {
    /// (node_id, probability) under softmax.
    pub probs: Vec<(NodeId, f32)>,
}

impl VocabDistribution {
    /// Build a softmax distribution over the active nodes in the ARG.
    ///
    /// Each node contributes its `attribution_score` as the logit.
    /// NodeId is the stable vocabulary index — not a transient edge ID.
    pub fn from_graph(graph: &ArgGraph, active_env: Env) -> Self {
        let active: Vec<(NodeId, f32)> = graph.node_indices()
            .filter(|&ni| {
                let n = &graph[ni];
                (n.atms_label & active_env) != 0
            })
            .map(|ni| {
                let n = &graph[ni];
                (n.id, n.attribution_score)
            })
            .collect();

        if active.is_empty() {
            return Self { probs: vec![] };
        }

        // Softmax over attribution scores.
        let max_score = active.iter().map(|&(_, s)| s).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = active.iter().map(|&(_, s)| (s - max_score).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>() + 1e-12;

        Self {
            probs: active.into_iter()
                .zip(exps)
                .map(|((nid, _), p)| (nid, p / sum))
                .collect(),
        }
    }

    /// Probability of the node with `node_id`.
    pub fn probability_of_node(&self, node_id: NodeId) -> f32 {
        self.probs.iter()
            .find(|&&(nid, _)| nid == node_id)
            .map(|&(_, p)| p)
            .unwrap_or(0.0)
    }

    /// Probability of the node with `node_id` (alias for probability_of_node).
    pub fn probability_for(&self, node_id: NodeId) -> f32 {
        self.probs.iter()
            .find(|&&(nid, _)| nid == node_id)
            .map(|&(_, p)| p)
            .unwrap_or(0.0)
    }

    /// CE quality split against the expected node.
    ///
    /// Returns:
    /// - `q_correct`: positive quality for the expected node's incoming edges
    ///   (pull toward correct). Signal = (1 - P(expected)).
    /// - `q_wrong`: if the top-scoring node differs from expected, a negative
    ///   quality for that node's incoming edges (push away from wrong).
    ///   Signal = -P(wrong).
    ///
    /// Caller applies positive update to expected node's incoming edges and
    /// negative update to the wrongly-predicted node's incoming edges (if any).
    pub fn ce_quality_split(&self, expected_node_id: NodeId) -> (Quality, Option<(NodeId, Quality)>) {
        let p_expected = self.probability_of_node(expected_node_id).max(1e-12);
        let predicted = self.probs.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(nid, p)| (nid, p));

        let q_correct = Quality::from_ce(p_expected, true);
        let q_wrong = predicted
            .filter(|&(nid, _)| nid != expected_node_id)
            .map(|(nid, p)| (nid, Quality::from_ce(p, false)));

        (q_correct, q_wrong)
    }
}

/// Stable node hash via FNV-1a (matches token_types::stable_node_id).
fn stable_node_id(predicate: &str) -> NodeId {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME:  u64 = 0x0000_0100_0000_01b3;
    predicate.bytes().fold(OFFSET, |h, b| h.wrapping_mul(PRIME) ^ b as u64)
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
        let total: f32 = dist.probs.iter().map(|&(_, p)| p).sum();
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
        let p_n1 = dist.probs.iter().find(|&&(nid, _)| nid == 1).map(|&(_, p)| p).unwrap_or(0.0);
        let q = Quality::from_ce(p_n1, true);
        assert!(q.as_f32() > 0.5,
            "low-prob correct prediction gives large CE update signal (1 - p ≈ high), got {}", q.as_f32());
    }

    #[test]
    fn ce_quality_split_wrong_prediction_gives_negative_signal() {
        let mut g: ArgGraph = StableGraph::new();
        let mt = ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right);
        // Node 1 = expected (low score), Node 2 = wrong prediction (high score).
        let mut n1 = ArgNode::new(1, NodeClass::DEFAULT, mt, (0, 0));
        n1.attribution_score = 0.1;
        n1.atms_label = 0b1;
        let mut n2 = ArgNode::new(2, NodeClass::DEFAULT, mt, (0, 0));
        n2.attribution_score = 10.0;
        n2.atms_label = 0b1;
        g.add_node(n1);
        g.add_node(n2);
        let dist = VocabDistribution::from_graph(&g, 0b1);
        let (q_correct, q_wrong) = dist.ce_quality_split(1);
        assert!(q_correct.as_f32() > 0.0, "expected node gets positive signal");
        let (wrong_nid, q_neg) = q_wrong.expect("should have a wrong-prediction penalty");
        assert_eq!(wrong_nid, 2, "wrong node is the high-score one");
        assert!(q_neg.is_negative(), "wrong prediction gets negative signal, got {}", q_neg.as_f32());
    }
}
