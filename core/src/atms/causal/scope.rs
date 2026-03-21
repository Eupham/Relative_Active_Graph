//! Counterfactual subgraph scoping for BF-ATMS.
//! The do(e=absent) intervention operates on a bounded subgraph to keep
//! BF-ATMS's polynomial guarantee intact.

use std::collections::{HashSet, VecDeque};
use crate::types::{NodeId, EdgeId};

/// A bounded subgraph over which BF-ATMS operates for one counterfactual query.
/// Contains at most `budget` nodes reachable from the intervened edge.
#[derive(Debug, Clone)]
pub struct CounterfactualScope {
    pub intervened_edge: EdgeId,
    pub nodes:          HashSet<NodeId>,
    pub edges:          HashSet<EdgeId>,
    pub budget:         usize,
}

impl CounterfactualScope {
    /// Build a scope by BFS from the source node of `intervened_edge`,
    /// limited to `budget` nodes.
    /// `adj`: adjacency function NodeId → [(neighbor NodeId, EdgeId)]
    pub fn build<F>(
        intervened_edge: EdgeId,
        root: NodeId,
        budget: usize,
        adj: F,
    ) -> Self
    where
        F: Fn(NodeId) -> Vec<(NodeId, EdgeId)>,
    {
        let mut nodes = HashSet::new();
        let mut edges = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(root);
        nodes.insert(root);

        while let Some(n) = queue.pop_front() {
            if nodes.len() >= budget { break; }
            for (neighbor, eid) in adj(n) {
                if eid == intervened_edge { continue; } // the removed edge
                if edges.insert(eid) && nodes.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        Self { intervened_edge, nodes, edges, budget }
    }

    pub fn contains_node(&self, id: NodeId) -> bool {
        self.nodes.contains(&id)
    }

    pub fn contains_edge(&self, id: EdgeId) -> bool {
        self.edges.contains(&id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_bounded_by_budget() {
        // Chain graph: 0→1→2→3→4→5, intervened edge = edge between 3 and 4
        let adj = |n: NodeId| -> Vec<(NodeId, EdgeId)> {
            if n < 5 { vec![(n + 1, n * 10)] } else { vec![] }
        };
        let scope = CounterfactualScope::build(30, 0, 3, adj);
        assert!(scope.node_count() <= 3);
    }

    #[test]
    fn intervened_edge_excluded() {
        let adj = |n: NodeId| -> Vec<(NodeId, EdgeId)> {
            match n {
                0 => vec![(1, 10), (2, 20)],
                1 => vec![(3, 30)],
                _ => vec![],
            }
        };
        // Intervene on edge 10 (0→1)
        let scope = CounterfactualScope::build(10, 0, 10, adj);
        assert!(!scope.contains_edge(10));
        assert!(scope.contains_edge(20)); // 0→2 is still there
    }
}
