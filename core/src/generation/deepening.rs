//! Progressive deepening: if hypotheses don't satisfy the query, deepen up to MAX_DEPTH times.
//! Each deepening round expands the ARG one step further and re-generates hypotheses.

use crate::types::{TRDId, ModalType, ModalMode, TypeCategory};
use crate::arg::ArgGraph;
use crate::adaptive::{PerfRegistry, ThresholdRegistry};
use crate::generation::{
    hypothesis::{Hypothesis, generate_hypotheses},
    traverser::ArgTraverser,
};
use crate::semantics::mtlg_semantics::MtlgSemantics;
use petgraph::visit::EdgeRef;
use std::collections::HashSet;

pub const MAX_DEPTH: usize = 3;

/// Result of one deepening round.
#[derive(Debug)]
pub struct DeepeningResult {
    pub hypotheses:  Vec<Hypothesis>,
    pub depth_used:  usize,
    pub satisfied:   bool,
}

/// Progressive deepening orchestrator.
pub struct ProgressiveDeepener {
    pub max_depth:  usize,
    pub active_trd: Option<TRDId>,
}

impl ProgressiveDeepener {
    pub fn new(active_trd: Option<TRDId>) -> Self {
        Self { max_depth: MAX_DEPTH, active_trd }
    }

    /// IDA*-inspired bounded DFS over the semantic graph.
    /// Current heuristic uses inverse attribution score as a proxy cost.
    /// Note: this is not full canonical IDA* with an admissible/consistent heuristic proof.
    pub fn run(
        &self,
        graph:         &ArgGraph,
        semantics:     &MtlgSemantics,
        perf:          &PerfRegistry,
        thresholds:    &mut ThresholdRegistry,
        _query_text:    &str,
        expected_type: &ModalType,
    ) -> DeepeningResult {
        let mut bound = 1.0; // Initial cost horizon bound for formal graph expansion
        
        for depth in 0..self.max_depth {
            // Dynamically expand structural environment boundary
            let mut traverser = ArgTraverser::new(50 * (depth + 1), self.active_trd);
            traverser.traverse(graph, perf, thresholds);

            let start_nodes: Vec<_> = graph.node_indices().filter(|&n| {
                graph.edges_directed(n, petgraph::Direction::Incoming).count() == 0
            }).collect();
            let start_nodes = if start_nodes.is_empty() { graph.node_indices().collect() } else { start_nodes };

            let mut path = Vec::new();
            let mut visited = HashSet::new();
            let mut satisfying_hyps = Vec::new();
            
            for start in start_nodes {
                let cost = self.ida_star_search(
                    graph, start, 0.0, bound, expected_type, 
                    semantics, &mut path, &mut visited, &mut satisfying_hyps
                );
                if cost == f32::NEG_INFINITY && !satisfying_hyps.is_empty() {
                    return DeepeningResult { hypotheses: satisfying_hyps, depth_used: depth, satisfied: true };
                }
                // Relax search bound when the current horizon is exhausted.
                bound = cost.max(bound + 1.0);
            }

            if depth == self.max_depth - 1 {
                return DeepeningResult { hypotheses: generate_hypotheses(graph, semantics, self.active_trd, perf), depth_used: depth, satisfied: false };
            }
        }
        DeepeningResult { hypotheses: vec![], depth_used: self.max_depth, satisfied: false }
    }

    /// DFS bounded by the current deepening cost horizon.
    /// `h(n)` uses inverse attribution as a practical ranking heuristic.
    fn ida_star_search(
        &self,
        graph: &ArgGraph,
        node: petgraph::graph::NodeIndex,
        g_cost: f32,
        bound: f32,
        expected_type: &ModalType,
        semantics: &MtlgSemantics,
        path: &mut Vec<petgraph::graph::NodeIndex>,
        visited: &mut HashSet<petgraph::graph::NodeIndex>,
        satisfying: &mut Vec<Hypothesis>
    ) -> f32 {
        let arg_node = &graph[node];
        let h_cost = 1.0 / (arg_node.attribution_score + 1e-6);
        let f_cost = g_cost + h_cost;
        
        if f_cost > bound { return f_cost; }

        let root_type = semantics.type_map.get(arg_node.surface_str().unwrap_or("")).copied()
            .unwrap_or(ModalType::atom(ModalMode::Diamond, TypeCategory::DEFAULT));
            
        let cat_match = expected_type.category == TypeCategory::DEFAULT 
            || root_type.category == TypeCategory::DEFAULT 
            || root_type.category == expected_type.category;
            
        if root_type.mode == expected_type.mode && cat_match {
            satisfying.push(Hypothesis::from_node(satisfying.len(), arg_node, semantics, 1.0 / f_cost));
            return f32::NEG_INFINITY; // Signifies logical derivation target reached
        }

        let mut min_bound = f32::INFINITY;
        path.push(node);
        visited.insert(node);

        for edge in graph.edges(node) {
            let neighbor = edge.target();
            if !visited.contains(&neighbor) {
                let t = self.ida_star_search(
                    graph, neighbor, g_cost + 1.0, bound, 
                    expected_type, semantics, path, visited, satisfying
                );
                if t == f32::NEG_INFINITY { return f32::NEG_INFINITY; }
                if t < min_bound { min_bound = t; }
            }
        }

        path.pop();
        visited.remove(&node);
        min_bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeClass};
    use crate::types::{ModalType, ModalMode, TypeCategory, Direction};
    use petgraph::stable_graph::StableGraph;
    use crate::arg::search::ArgGraph;

    fn node_with_surface(id: u64, surface: &str, score: f32) -> ArgNode {
        let mut n = ArgNode::new(id, NodeClass::DEFAULT,
            ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right), (0,0));
        n.surface = Some(surface.as_bytes().to_vec());
        n.attribution_score = score;
        n.atms_label = 0b1;
        n
    }

    #[test]
    fn finds_satisfying_at_depth_0() {
        let mut g: ArgGraph = StableGraph::new();
        // use high attribution score so that 1.0 / (score + 1e-6) <= 1.0 initial bound
        g.add_node(node_with_surface(1, "run", 1.0));
        let perf       = PerfRegistry::new(0.25);
        let mut thresh = ThresholdRegistry::default();
        let deepener   = ProgressiveDeepener::new(None);

        // The node has Diamond/Scene type; register it in the semantics type_map
        // by using a semantics with "run" → Diamond/Scene.
        let mut sem2 = MtlgSemantics::new();
        sem2.register_type(
            "run".into(),
            ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right),
        );
        let expected = ModalType::atom(ModalMode::Diamond, TypeCategory::DEFAULT);
        let result = deepener.run(&g, &sem2, &perf, &mut thresh, "run", &expected);
        assert!(result.satisfied, "should find 'run' at depth 0");
        assert_eq!(result.depth_used, 0);
    }

    #[test]
    fn exhausts_depth_without_match() {
        let mut g: ArgGraph = StableGraph::new();
        g.add_node(node_with_surface(1, "run", 0.9));
        let sem        = MtlgSemantics::new();
        let perf       = PerfRegistry::new(0.25);
        let mut thresh = ThresholdRegistry::default();
        let deepener   = ProgressiveDeepener::new(None);

        // Query expects Box/Process — node is Diamond/Scene — no match.
        let expected = ModalType::atom(ModalMode::Box, TypeCategory(1));
        let result = deepener.run(&g, &sem, &perf, &mut thresh, "nonexistent_predicate", &expected);
        assert!(!result.satisfied);
    }
}
