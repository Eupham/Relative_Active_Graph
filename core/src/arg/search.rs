//! Lazy ARG expansion: activate nodes where s ⊨ α(v) > θ_α(TRD, t).
//! Budget-bounded: at most MAX_NODES_PER_CONTEXT nodes per G(s).
//! Uses a priority queue ordered by attribution_score × trd_relevance.

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use petgraph::stable_graph::{StableGraph, NodeIndex};
use petgraph::Directed;
use crate::types::{NodeId, EdgeId, Env, TRDId, ModalType, ModalMode};
use crate::arg::{
    node::{ArgNode, NodeClass},
    edge::{ArgEdge, EdgeClass},
};
use crate::atms::base::env::subsumes;

pub const MAX_NODES_PER_CONTEXT: usize = 500;

/// The active relative graph G(s): the portion of the potential graph that's
/// activated in the current situation context.
pub type ArgGraph = StableGraph<ArgNode, ArgEdge, Directed>;

/// Priority entry for lazy activation.
#[derive(Clone, Debug)]
struct ActivationCandidate {
    score:   f32,
    node_id: NodeId,
}

impl PartialEq for ActivationCandidate {
    fn eq(&self, other: &Self) -> bool { self.score == other.score }
}
impl Eq for ActivationCandidate {}
impl PartialOrd for ActivationCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.score.partial_cmp(&other.score) }
}
impl Ord for ActivationCandidate {
    fn cmp(&self, other: &Self) -> Ordering { self.partial_cmp(other).unwrap_or(Ordering::Equal) }
}

/// ARG search/expansion state for a single context.
pub struct ArgSearch {
    pub graph: ArgGraph,
    /// Map from NodeId to petgraph NodeIndex.
    node_index: fnv::FnvHashMap<NodeId, NodeIndex>,
    active_env: Env,
    theta_alpha: f64,
    theta_rho:   f64,
    pub node_count: usize,
}

impl ArgSearch {
    pub fn new(active_env: Env, theta_alpha: f64, theta_rho: f64) -> Self {
        Self {
            graph:       StableGraph::new(),
            node_index:  fnv::FnvHashMap::default(),
            active_env,
            theta_alpha,
            theta_rho,
            node_count: 0,
        }
    }

    /// Attempt to activate `node` in the current context.
    /// Returns true if the node was newly activated.
    pub fn try_activate(&mut self, node: ArgNode) -> bool {
        if self.node_count >= MAX_NODES_PER_CONTEXT { return false; }
        if !node.is_active(self.active_env, self.theta_alpha) { return false; }
        if self.node_index.contains_key(&node.id) { return false; }
        let idx = self.graph.add_node(node.clone());
        self.node_index.insert(node.id, idx);
        self.node_count += 1;
        true
    }

    /// Add an edge if both endpoints are active and edge weight exceeds theta_rho.
    pub fn try_add_edge(&mut self, edge: ArgEdge) -> bool {
        if edge.weight as f64 <= self.theta_rho { return false; }
        let src_idx = match self.node_index.get(&edge.src) { Some(&i) => i, None => return false };
        let dst_idx = match self.node_index.get(&edge.dst) { Some(&i) => i, None => return false };
        self.graph.add_edge(src_idx, dst_idx, edge);
        true
    }

    /// Expand the graph greedily from a set of seed candidates.
    /// Each candidate is (node, edges_from_node).
    pub fn expand<F>(&mut self, seed_ids: &[NodeId], fetch: &mut F)
    where
        F: FnMut(NodeId) -> Option<(ArgNode, Vec<(ArgEdge, ArgNode)>)>,
    {
        let mut queue: BinaryHeap<ActivationCandidate> = seed_ids.iter().map(|&id| {
            ActivationCandidate { score: 0.5, node_id: id }
        }).collect();
        let mut visited: HashSet<NodeId> = HashSet::new();

        while let Some(cand) = queue.pop() {
            if visited.contains(&cand.node_id) { continue; }
            visited.insert(cand.node_id);
            if self.node_count >= MAX_NODES_PER_CONTEXT { break; }

            if let Some((node, neighbors)) = fetch(cand.node_id) {
                let score = node.attribution_score;
                self.try_activate(node);
                for (edge, neighbor) in neighbors {
                    if self.try_activate(neighbor.clone()) {
                        self.try_add_edge(edge);
                        queue.push(ActivationCandidate {
                            score: neighbor.attribution_score,
                            node_id: neighbor.id,
                        });
                    }
                }
            }
        }
    }

    pub fn node_idx(&self, id: NodeId) -> Option<NodeIndex> {
        self.node_index.get(&id).copied()
    }

    pub fn active_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.node_index.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalType, ModalMode, TypeCategory};

    fn make_active_node(id: NodeId, env: Env, score: f32) -> ArgNode {
        let mut n = ArgNode::new(id, NodeClass::DEFAULT, ModalType::default(), (0, 0));
        n.atms_label = env;
        n.attribution_score = score;
        n
    }

    #[test]
    fn activates_above_threshold() {
        let mut search = ArgSearch::new(0b11, 0.3, 0.2);
        let n = make_active_node(1, 0b01, 0.5);
        assert!(search.try_activate(n));
        assert_eq!(search.node_count, 1);
    }

    #[test]
    fn rejects_below_threshold() {
        let mut search = ArgSearch::new(0b11, 0.7, 0.2);
        let n = make_active_node(1, 0b01, 0.4); // score < theta
        assert!(!search.try_activate(n));
    }
}
