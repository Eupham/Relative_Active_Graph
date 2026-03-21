//! Lazy ARG traversal with TRD-aware budget.
//! Traverses G(s) in scheduling-priority order (b/t-level), respecting the
//! MAX_NODES_PER_CONTEXT budget and TRD-relative activation thresholds.

use std::collections::HashSet;
use crate::types::{NodeId, TRDId, Env};
use crate::arg::{ArgGraph, ArgNode};
use crate::scheduler::dependency::priority_order;
use crate::adaptive::{PerfRegistry, ThresholdRegistry};
use petgraph::visit::EdgeRef;

/// A traversal visit record.
#[derive(Debug, Clone)]
pub struct TraversalStep {
    pub node_id:   NodeId,
    pub depth:     f32,
    pub relevance: f32,
}

/// Lazy ARG traverser: walks G(s) in priority order, collecting derivation steps.
pub struct ArgTraverser {
    visited:     HashSet<NodeId>,
    pub steps:   Vec<TraversalStep>,
    budget:      usize,
    active_trd:  Option<TRDId>,
}

impl ArgTraverser {
    pub fn new(budget: usize, active_trd: Option<TRDId>) -> Self {
        Self {
            visited:    HashSet::new(),
            steps:      Vec::new(),
            budget,
            active_trd,
        }
    }

    /// Traverse the graph, collecting steps in priority order.
    pub fn traverse(
        &mut self,
        graph:     &ArgGraph,
        perf:      &PerfRegistry,
        thresholds: &mut ThresholdRegistry,
    ) {
        let priorities = priority_order(graph);
        let theta_alpha = self.active_trd
            .map(|trd| thresholds.theta_alpha(trd))
            .unwrap_or(0.4);

        for (node_id, priority) in priorities {
            if self.steps.len() >= self.budget { break; }
            if self.visited.contains(&node_id) { continue; }

            // Find the node in the graph.
            let Some(idx) = graph.node_indices().find(|&i| graph[i].id == node_id) else {
                continue;
            };
            let node = &graph[idx];

            // TRD-relative activation check.
            if (node.attribution_score as f64) < theta_alpha { continue; }

            self.visited.insert(node_id);
            let relevance = self.active_trd
                .map(|trd| if node.trd_membership.contains(&trd) { perf.p_ema(trd) as f32 } else { 0.3 })
                .unwrap_or(0.5);

            self.steps.push(TraversalStep {
                node_id,
                depth:     node.depth,
                relevance,
            });
        }
    }

    /// Get ordered node IDs from traversal.
    pub fn traversal_order(&self) -> Vec<NodeId> {
        self.steps.iter().map(|s| s.node_id).collect()
    }

    /// Neighbors reachable from `node_id` in the graph (used for deepening).
    pub fn reachable_from(graph: &ArgGraph, node_id: NodeId) -> Vec<NodeId> {
        let Some(idx) = graph.node_indices().find(|&i| graph[i].id == node_id) else {
            return vec![];
        };
        graph.edges(idx).map(|e| graph[e.target()].id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeType, ArgEdge, EdgeType};
    use crate::types::{ModalType, ModalMode};
    use petgraph::stable_graph::StableGraph;

    fn node(id: u64, score: f32) -> ArgNode {
        let mut n = ArgNode::new(id, NodeType::Concept, ModalType::default(), (0,0));
        n.attribution_score = score;
        n.atms_label = 0b1;
        n
    }

    #[test]
    fn traverses_in_order() {
        let mut g: ArgGraph = StableGraph::new();
        let a = g.add_node(node(1, 0.9));
        let b = g.add_node(node(2, 0.5));
        g.add_edge(a, b, ArgEdge::new(1, 1, 2, EdgeType::Composition, ModalMode::Diamond));

        let mut traverser = ArgTraverser::new(10, None);
        let mut thresholds = ThresholdRegistry::default();
        let perf = PerfRegistry::new(0.25);
        traverser.traverse(&g, &perf, &mut thresholds);

        let order = traverser.traversal_order();
        assert!(!order.is_empty());
    }
}
