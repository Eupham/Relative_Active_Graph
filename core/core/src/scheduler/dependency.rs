//! b/t-level computation over G(s) for list scheduling.
//! b-level (bottom level): longest path from node to any sink.
//! t-level (top level): longest path from any source to node.
//! Parallelism: nodes at the same b/t-level can execute simultaneously.

use std::collections::HashMap;
use crate::types::NodeId;
use crate::arg::ArgGraph;
use petgraph::visit::EdgeRef;
use petgraph::algo::toposort;
use petgraph::Direction;

/// Compute b-levels (bottom-level = longest path to sink, weighted by node depth).
pub fn compute_b_levels(graph: &ArgGraph) -> HashMap<NodeId, f32> {
    let sorted = match toposort(graph, None) {
        Ok(order) => order,
        Err(_) => {
            // Cycle detected — return 0 for all nodes.
            return graph.node_indices().map(|i| (graph[i].id, 0.0)).collect();
        }
    };

    let mut b_level: HashMap<NodeId, f32> = HashMap::new();

    // Process in reverse topological order (sinks first).
    for &idx in sorted.iter().rev() {
        let node = &graph[idx];
        let self_cost = node.depth.max(0.001); // depth as cost
        let max_child_b = graph.edges_directed(idx, Direction::Outgoing)
            .map(|e| b_level.get(&graph[e.target()].id).copied().unwrap_or(0.0))
            .fold(0.0f32, f32::max);
        b_level.insert(node.id, self_cost + max_child_b);
    }
    b_level
}

/// Compute t-levels (top-level = longest path from source to this node).
pub fn compute_t_levels(graph: &ArgGraph) -> HashMap<NodeId, f32> {
    let sorted = match toposort(graph, None) {
        Ok(order) => order,
        Err(_) => return graph.node_indices().map(|i| (graph[i].id, 0.0)).collect(),
    };

    let mut t_level: HashMap<NodeId, f32> = HashMap::new();

    for &idx in sorted.iter() {
        let node = &graph[idx];
        let self_cost = node.depth.max(0.001);
        let max_parent_t = graph.edges_directed(idx, Direction::Incoming)
            .map(|e| t_level.get(&graph[e.source()].id).copied().unwrap_or(0.0))
            .fold(0.0f32, f32::max);
        t_level.insert(node.id, max_parent_t + self_cost);
    }
    t_level
}

/// Combined scheduling priority = b_level × (1 / (1 + t_level)).
/// Nodes with high b-level (many successors) and low t-level (close to root) run first.
pub fn scheduling_priority(b: f32, t: f32) -> f32 {
    b / (1.0 + t)
}

/// Return nodes sorted by scheduling priority (highest first = run first).
pub fn priority_order(graph: &ArgGraph) -> Vec<(NodeId, f32)> {
    let b_levels = compute_b_levels(graph);
    let t_levels = compute_t_levels(graph);
    let mut order: Vec<(NodeId, f32)> = graph.node_indices().map(|idx| {
        let id = graph[idx].id;
        let b  = b_levels.get(&id).copied().unwrap_or(0.0);
        let t  = t_levels.get(&id).copied().unwrap_or(0.0);
        (id, scheduling_priority(b, t))
    }).collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeClass, ArgEdge, EdgeClass};
    use crate::types::{ModalType, ModalMode, TypeCategory};
    use petgraph::stable_graph::StableGraph;

    fn node(id: u64, depth: f32) -> ArgNode {
        let mut n = ArgNode::new(id, NodeClass::DEFAULT, ModalType::default(), (0,0));
        n.depth = depth;
        n.atms_label = 0b1;
        n
    }

    #[test]
    fn b_level_increases_toward_sources() {
        let mut g: ArgGraph = StableGraph::new();
        let a = g.add_node(node(1, 1.0));
        let b = g.add_node(node(2, 1.0));
        let c = g.add_node(node(3, 1.0));
        g.add_edge(a, b, ArgEdge::new(1, 1, 2, EdgeClass::DEFAULT, ModalMode::Diamond));
        g.add_edge(b, c, ArgEdge::new(2, 2, 3, EdgeClass::DEFAULT, ModalMode::Diamond));

        let b_levels = compute_b_levels(&g);
        // a → b → c: b_level(a) > b_level(b) > b_level(c)
        assert!(b_levels[&1] > b_levels[&2], "a should have higher b-level than b");
        assert!(b_levels[&2] > b_levels[&3], "b should have higher b-level than c");
    }
}
