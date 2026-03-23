//! do(e=absent) intervention via BF-ATMS subgraph.
//!
//! Combines SCM intervention with BF-ATMS counterfactual scope.
//! The SCM intervention targets the *child* node of the removed edge
//! (Pearl do-calculus: cut the incoming structural equation of dst, not zero src).

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId, Env};
use crate::atms::causal::{BfAtms, CounterfactualScope};
use crate::atms::base::env::singleton;
use crate::causal::scm::Scm;
use crate::arg::{ArgGraph, ArgEdge};
use petgraph::visit::EdgeRef;

#[derive(Debug)]
pub struct InterventionResult {
    pub edge_id:          EdgeId,
    pub causal_fraction:  f64,
    pub quality_delta:    f64,
    /// NOGOOD to inject if causal_fraction > 0.5 (the intervention creates contradiction).
    pub nogood:           Option<Env>,
}

fn build_bf_atms(
    graph:      &ArgGraph,
    scope:      &CounterfactualScope,
    active_env: Env,
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

/// Resolve the ATMS assumption bits that underlie `edge_id`.
///
/// Per §4c: reads edge.label directly (the minimal assumption set for this edge,
/// populated at edge creation from the ATMS justification head's environment).
/// Seed edges have label = 0 (Env::EMPTY). The old src_label & !dst_label
/// heuristic is removed entirely.
fn resolve_edge_assumptions(graph: &ArgGraph, edge_id: EdgeId) -> Env {
    graph
        .edge_indices()
        .find(|&ei| graph[ei].id == edge_id)
        .map(|ei| graph[ei].label)
        .unwrap_or(0)
}

/// Execute a do(e=absent) intervention on `edge_id`.
///
/// Causal delta is measured at the *destination* (child) node of the edge.
/// The SCM intervention removes dst's incoming equation from src (Pearl §3.2),
/// replacing it with the observed mean of dst in the current context.
pub fn do_absent(
    graph:      &ArgGraph,
    scm:        &Scm,
    edge_id:    EdgeId,
    active_env: Env,
    budget:     usize,
) -> InterventionResult {
    let edge_data = graph.edge_indices()
        .find(|&ei| graph[ei].id == edge_id)
        .map(|ei| {
            let (src_idx, dst_idx) = graph.edge_endpoints(ei).unwrap();
            (graph[src_idx].id, graph[dst_idx].id)
        });

    let (src_id, dst_id) = match edge_data {
        Some(d) => d,
        None    => return InterventionResult {
            edge_id, causal_fraction: 0.0, quality_delta: 0.0, nogood: None,
        },
    };

    // BFS scope rooted at dst (the node losing the incoming edge).
    let adj = |nid: NodeId| -> Vec<(NodeId, EdgeId)> {
        graph.node_indices()
            .find(|&i| graph[i].id == nid)
            .map(|idx| graph.edges(idx).map(|e| (graph[e.target()].id, e.weight().id)).collect())
            .unwrap_or_default()
    };
    let scope = CounterfactualScope::build(edge_id, dst_id, budget, adj);
    let edge_assumptions = resolve_edge_assumptions(graph, edge_id);

    let mut bf = build_bf_atms(graph, &scope, active_env);
    let result = bf.run_intervention(edge_assumptions, active_env);

    // SCM quality delta: measured at dst (child), not src (§3).
    // Baseline: computed value under current structural equations (not observed value).
    let scm_before = scm.compute(dst_id).unwrap_or(0.5);

    // Sever src → dst by zeroing src's coefficient in dst's structural equation.
    // Pearl do-calculus: do(X=absent) ≡ setting all β_k = 0 for parent X (§3b).
    let counterfactual_scm = scm.remove_parent(dst_id, src_id);
    let scm_after = counterfactual_scm.compute(dst_id).unwrap_or(scm_before);

    let quality_delta = scm_before - scm_after;
    // Aggregation: direct counterfactual delta on dst_id only.
    // Downstream propagation (§3c) is deferred; delta on dst captures the primary effect.

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
    use petgraph::stable_graph::StableGraph;

    #[test]
    fn missing_edge_returns_zero() {
        let g: ArgGraph = StableGraph::new();
        let scm = Scm::new();
        let result = do_absent(&g, &scm, 999, 0b1, 10);
        assert_eq!(result.causal_fraction, 0.0);
        assert_eq!(result.quality_delta, 0.0);
    }
}
