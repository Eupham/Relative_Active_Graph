//! Apply attribution deltas to ARG edge weights.
//! w(e) += η · Δ(e) · Q   on each TR dissolution.
//! Global invariant: the ARG is modified ONLY by dissolved TR attribution traces.

use std::collections::HashMap;
use crate::types::{EdgeId, NodeId, Quality};
use crate::arg::ArgGraph;
use crate::feedback::attribution::EdgeAttribution;

/// Learning rate η.
pub const ETA: f32 = 0.05;

/// Apply a single attribution delta to the graph's edge weight.
pub fn apply_attribution(
    graph:   &mut ArgGraph,
    edge_id: EdgeId,
    delta:   f64,
    quality: Quality,
) -> bool {
    let Some(ei) = graph.edge_indices().find(|&i| graph[i].id == edge_id) else {
        return false;
    };
    let update = ETA * (delta as f32) * quality.as_f32();
    graph[ei].apply_delta(update);
    true
}

/// Apply all attribution deltas from a dissolved TR to the ARG.
/// Returns the number of edges updated.
pub fn apply_tr_attribution(
    graph:        &mut ArgGraph,
    attributions: &[EdgeAttribution],
    quality:      Quality,
) -> usize {
    attributions.iter().filter(|attr| {
        apply_attribution(graph, attr.edge_id, attr.delta, quality)
    }).count()
}

/// Apply attribution trace from a TR directly (from transient_repr attribution_trace map).
pub fn apply_trace(
    graph:        &mut ArgGraph,
    trace:        &HashMap<EdgeId, f32>,
    quality:      Quality,
) -> usize {
    let q = quality.as_f32();
    let mut updated = 0;
    for (&edge_id, &contrib) in trace {
        if let Some(ei) = graph.edge_indices().find(|&i| graph[i].id == edge_id) {
            let update = ETA * contrib * q;
            graph[ei].apply_delta(update);
            updated += 1;
        }
    }
    updated
}

/// Decay all edge weights slightly toward 0.5 (neutral prior) to prevent entrenchment.
/// Called periodically (e.g., every N TR dissolutions).
pub fn apply_weight_decay(graph: &mut ArgGraph, decay_rate: f32) {
    let edges: Vec<_> = graph.edge_indices().collect();
    for ei in edges {
        let w = graph[ei].weight;
        graph[ei].weight = w + decay_rate * (0.5 - w);
    }
}

/// Backward attribution through the ATMS justification chain.
///
/// When a generated token at depth 0 produces a CE quality signal, that signal
/// propagates backward through the justifications that derived the contributing
/// nodes — replacing backpropagation's chain rule.
///
/// `start_nodes`: NodeIds whose incoming edges should receive the initial signal.
/// `quality`:     CE-equivalent quality (1.0 - p_correct for correct path).
/// `max_depth`:   Justification levels to traverse (4 covers most derivations).
/// `decay`:       Signal decay per level (0.7 keeps signal meaningful to depth 3).
pub fn propagate_attribution_backward(
    graph:       &mut ArgGraph,
    atms:        &crate::atms::BaseAtms,
    start_nodes: &[NodeId],
    quality:     Quality,
    max_depth:   usize,
    decay:       f32,
) {
    let mut frontier: Vec<(NodeId, f32)> =
        start_nodes.iter().map(|&n| (n, quality.as_f32())).collect();

    for _depth in 0..max_depth {
        if frontier.is_empty() { break; }
        let mut next = Vec::new();

        for (node_id, signal) in &frontier {
            let decayed = signal * decay;
            if decayed < 0.005 { continue; }

            // Update all incoming ARG edges for this node.
            let incoming: Vec<EdgeId> = graph.edge_indices()
                .filter(|&ei| graph[ei].dst == *node_id)
                .map(|ei| graph[ei].id)
                .collect();

            for eid in &incoming {
                apply_attribution(graph, *eid, 1.0, Quality::new(decayed));
            }

            // Follow backward through ATMS justification chain.
            for antecedent in atms.antecedents_of(*node_id) {
                next.push((antecedent, decayed));
            }
        }
        frontier = next;
    }
}

/// Apply negative attribution to all incoming edges of nodes whose ATMS labels
/// have become inconsistent due to a newly added NOGOOD.
///
/// This closes the loop: NOGOOD detected → TR dissolution → negative attribution.
/// Without this, wrong paths are never actively weakened; the system can only
/// be pushed toward correct answers, not pulled away from wrong ones.
pub fn apply_nogood_consequences(
    graph:            &mut ArgGraph,
    inconsistent_ids: &[NodeId],
) {
    for &node_id in inconsistent_ids {
        let edge_ids: Vec<EdgeId> = graph.edge_indices()
            .filter(|&ei| graph[ei].dst == node_id)
            .map(|ei| graph[ei].id)
            .collect();
        for eid in edge_ids {
            apply_attribution(graph, eid, 1.0, Quality::BAD);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeClass, ArgEdge, EdgeClass};
    use crate::types::{ModalType, ModalMode};
    use petgraph::stable_graph::StableGraph;

    fn setup_graph_with_edge() -> ArgGraph {
        let mut g: ArgGraph = StableGraph::new();
        let mut n1 = ArgNode::new(1, NodeClass::DEFAULT, ModalType::default(), (0,0));
        let mut n2 = ArgNode::new(2, NodeClass::DEFAULT, ModalType::default(), (0,0));
        n1.atms_label = 0b1; n2.atms_label = 0b1;
        let i1 = g.add_node(n1);
        let i2 = g.add_node(n2);
        let mut e = ArgEdge::new(42, 1, 2, EdgeClass::DEFAULT, ModalMode::Diamond);
        e.weight = 0.5;
        g.add_edge(i1, i2, e);
        g
    }

    #[test]
    fn attribution_increases_weight() {
        let mut g = setup_graph_with_edge();
        let before = g.edge_indices().next().map(|ei| g[ei].weight).unwrap();
        apply_attribution(&mut g, 42, 1.0, Quality::GOOD);
        let after = g.edge_indices().next().map(|ei| g[ei].weight).unwrap();
        assert!(after > before, "weight should increase after good attribution");
    }

    #[test]
    fn decay_moves_toward_neutral() {
        let mut g = setup_graph_with_edge();
        // Set weight to 0.8 (above neutral)
        if let Some(ei) = g.edge_indices().next() { g[ei].weight = 0.8; }
        apply_weight_decay(&mut g, 0.1);
        let w = g.edge_indices().next().map(|ei| g[ei].weight).unwrap();
        assert!(w < 0.8 && w > 0.5, "weight={w} should move toward 0.5");
    }
}
