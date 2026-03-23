//! TRD-relative cost function for scheduling.
//! cost(v, s) = depth(v, s) × (1 / relevance(TRD, t))
//! Higher cost → scheduled later. Low-cost nodes (high TRD relevance) run first.

use std::collections::HashMap;
use crate::types::{NodeId, TRDId};
use crate::arg::ArgGraph;
use crate::adaptive::PerfRegistry;
use super::dependency::priority_order;

/// Compute TRD relevance score for a node given the current TRD's performance.
/// Higher EMA performance → more relevant (we trust this TRD's structure more).
fn trd_relevance(node_trd_ids: &[TRDId], active_trd: Option<TRDId>, perf: &PerfRegistry) -> f32 {
    if node_trd_ids.is_empty() { return 0.5; }
    let base = match active_trd {
        Some(trd) if node_trd_ids.contains(&trd) => perf.p_ema(trd) as f32,
        _ => 0.3, // node not in active TRD: deprioritize
    };
    base.clamp(0.01, 1.0)
}

/// TRD-relative scheduling cost for a node.
pub fn node_cost(node_id: NodeId, graph: &ArgGraph, active_trd: Option<TRDId>, perf: &PerfRegistry) -> f32 {
    let idx = match graph.node_indices().find(|&i| graph[i].id == node_id) {
        Some(i) => i,
        None    => return f32::MAX,
    };
    let node      = &graph[idx];
    let relevance = trd_relevance(&node.trd_membership, active_trd, perf);
    // cost = depth / relevance: deep + irrelevant nodes are most expensive
    node.depth / relevance
}

/// A scheduled work item.
#[derive(Debug)]
pub struct ScheduledItem {
    pub node_id:  NodeId,
    pub cost:     f32,
    pub priority: f32,
}

/// Produce the full execution schedule for G(s), TRD-relative.
pub fn build_schedule(
    graph:       &ArgGraph,
    active_trd:  Option<TRDId>,
    perf:        &PerfRegistry,
    ready_nodes: &[NodeId],
) -> Vec<ScheduledItem> {
    let priority_map: HashMap<NodeId, f32> = priority_order(graph).into_iter().collect();

    let mut items: Vec<ScheduledItem> = ready_nodes.iter().map(|&nid| {
        let cost     = node_cost(nid, graph, active_trd, perf);
        let priority = priority_map.get(&nid).copied().unwrap_or(0.0);
        ScheduledItem { node_id: nid, cost, priority }
    }).collect();

    // Sort: highest priority first, breaking ties by lowest cost.
    items.sort_by(|a, b| {
        b.priority.partial_cmp(&a.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal))
    });
    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeClass, ArgEdge, EdgeClass};
    use crate::types::{ModalType, Quality};
    use petgraph::stable_graph::StableGraph;

    fn node_with_depth(id: u64, depth: f32, trd: TRDId) -> ArgNode {
        let mut n = ArgNode::new(id, NodeClass::DEFAULT, ModalType::default(), (0,0));
        n.depth = depth;
        n.trd_membership = vec![trd];
        n.atms_label = 0b1;
        n
    }

    #[test]
    fn higher_relevance_gets_lower_cost() {
        let mut g: ArgGraph = StableGraph::new();
        g.add_node(node_with_depth(1, 1.0, 0));
        g.add_node(node_with_depth(2, 1.0, 1));

        let mut perf = PerfRegistry::new(0.25);
        // TRD 0: all good (high p_ema)
        for _ in 0..10 { perf.update(0, Quality::GOOD); }
        // TRD 1: all bad (low p_ema)
        for _ in 0..10 { perf.update(1, Quality::BAD); }

        let cost0 = node_cost(1, &g, Some(0), &perf);
        let cost1 = node_cost(2, &g, Some(1), &perf);
        // Node in high-performance TRD should have lower cost.
        assert!(cost0 < cost1, "cost0={cost0} should < cost1={cost1}");
    }
}
