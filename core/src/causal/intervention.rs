//! do(e=absent) intervention via BF-ATMS subgraph.
//! Combines the SCM intervention with BF-ATMS counterfactual scope.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId, Env};
use crate::atms::causal::{BfAtms, CounterfactualScope};
use crate::causal::scm::Scm;
use crate::arg::{ArgGraph, ArgEdge};
use petgraph::visit::EdgeRef;

/// Result of a do(e=absent) intervention.
#[derive(Debug)]
pub struct InterventionResult {
    pub edge_id:          EdgeId,
    pub causal_fraction:  f64,
    pub quality_delta:    f64,
    /// NOGOOD to inject if intervention creates contradiction (causal_fraction > 0.5).
    pub nogood:           Option<Env>,
}

/// Build a BF-ATMS from the ARG graph restricted to the counterfactual scope.
fn build_bf_atms(
    graph:        &ArgGraph,
    scope:        &CounterfactualScope,
    active_env:   Env,
) -> BfAtms {
    let mut bf = BfAtms::new(scope.clone());
    for idx in graph.node_indices() {
        let node = &graph[idx];
        if scope.contains_node(node.id) {
            bf.seed_node(node.id, node.atms_label, node.attribution_score as f64);
        }
    }
    bf
}

/// Execute a do(e=absent) intervention on `edge_id`.
pub fn do_absent(
    graph:        &ArgGraph,
    scm:          &Scm,
    edge_id:      EdgeId,
    active_env:   Env,
    budget:       usize,
) -> InterventionResult {
    // Find the edge and its source node.
    let edge_data = graph.edge_indices()
        .find(|&ei| graph[ei].id == edge_id)
        .map(|ei| {
            let (src_idx, _) = graph.edge_endpoints(ei).unwrap();
            (graph[ei].src, src_idx)
        });

    let (src_id, src_idx) = match edge_data {
        Some(d) => d,
        None => return InterventionResult {
            edge_id, causal_fraction: 0.0, quality_delta: 0.0, nogood: None,
        },
    };

    // Build adjacency for BFS scope construction.
    let adj = |nid: NodeId| -> Vec<(NodeId, EdgeId)> {
        graph.node_indices()
            .find(|&i| graph[i].id == nid)
            .map(|idx| {
                graph.edges(idx).map(|e| (graph[e.target()].id, e.weight().id)).collect()
            })
            .unwrap_or_default()
    };

    let scope = CounterfactualScope::build(edge_id, src_id, budget, adj);
    let mut bf = build_bf_atms(graph, &scope, active_env);
    let result = bf.run_intervention(edge_id, active_env);

    // Also compute SCM quality delta: value at scope boundary with vs without intervention.
    let scm_before = scm.compute(src_id).unwrap_or(0.5);
    let intervened_scm = scm.intervene(src_id, 0.0); // do(src=0)
    let scm_after  = intervened_scm.compute(src_id).unwrap_or(0.0);
    let quality_delta = scm_before - scm_after;

    InterventionResult {
        edge_id,
        causal_fraction: result.causal_fraction,
        quality_delta,
        nogood: result.nogood,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeType, ArgEdge, EdgeType};
    use crate::types::{ModalType, ModalMode};
    use petgraph::stable_graph::StableGraph;

    #[test]
    fn missing_edge_returns_zero() {
        let g: ArgGraph = StableGraph::new();
        let scm = Scm::new();
        let result = do_absent(&g, &scm, 999, 0b1, 10);
        assert_eq!(result.causal_fraction, 0.0);
    }
}
