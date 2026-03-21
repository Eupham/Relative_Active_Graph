//! Apply attribution deltas to ARG edge weights.
//! w(e) += η · Δ(e) · Q   on each TR dissolution.
//! Global invariant: the ARG is modified ONLY by dissolved TR attribution traces.

use std::collections::HashMap;
use crate::types::{EdgeId, Quality};
use crate::arg::ArgGraph;
use crate::feedback::attribution::EdgeAttribution;
use petgraph::stable_graph::EdgeIndex;

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
    let update = ETA * (delta as f32) * (quality.as_f64() as f32);
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
    let q = quality.as_f64() as f32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeType, ArgEdge, EdgeType};
    use crate::types::{ModalType, ModalMode};
    use petgraph::stable_graph::StableGraph;

    fn setup_graph_with_edge() -> ArgGraph {
        let mut g: ArgGraph = StableGraph::new();
        let mut n1 = ArgNode::new(1, NodeType::Concept, ModalType::default(), (0,0));
        let mut n2 = ArgNode::new(2, NodeType::Concept, ModalType::default(), (0,0));
        n1.atms_label = 0b1; n2.atms_label = 0b1;
        let i1 = g.add_node(n1);
        let i2 = g.add_node(n2);
        let mut e = ArgEdge::new(42, 1, 2, EdgeType::Composition, ModalMode::Diamond);
        e.weight = 0.5;
        g.add_edge(i1, i2, e);
        g
    }

    #[test]
    fn attribution_increases_weight() {
        let mut g = setup_graph_with_edge();
        let before = g.edge_indices().next().map(|ei| g[ei].weight).unwrap();
        apply_attribution(&mut g, 42, 1.0, Quality::Good);
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
