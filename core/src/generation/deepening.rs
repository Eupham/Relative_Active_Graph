//! Progressive deepening: if hypotheses don't satisfy the query, deepen up to MAX_DEPTH times.
//! Each deepening round expands the ARG one step further and re-generates hypotheses.

use crate::types::{NodeId, TRDId};
use crate::arg::{ArgGraph, ArgNode, ArgEdge, NodeType, EdgeType};
use crate::adaptive::{PerfRegistry, ThresholdRegistry};
use crate::generation::{
    hypothesis::{Hypothesis, generate_hypotheses, filter_satisfying},
    traverser::ArgTraverser,
};
use crate::semantics::mtlg_semantics::MtlgSemantics;

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

    /// Attempt to find satisfying hypotheses, deepening up to `max_depth` times.
    pub fn run(
        &self,
        graph:      &ArgGraph,
        semantics:  &MtlgSemantics,
        perf:       &PerfRegistry,
        thresholds: &mut ThresholdRegistry,
        query:      &str,
    ) -> DeepeningResult {
        let budget_per_depth = 50; // nodes per traversal pass

        for depth in 0..self.max_depth {
            let mut traverser = ArgTraverser::new(budget_per_depth * (depth + 1), self.active_trd);
            traverser.traverse(graph, perf, thresholds);

            let hyps = generate_hypotheses(graph, semantics, self.active_trd, perf);
            let satisfying = filter_satisfying(hyps.clone(), query);

            if !satisfying.is_empty() {
                return DeepeningResult {
                    hypotheses: satisfying,
                    depth_used: depth,
                    satisfied:  true,
                };
            }

            // On final depth: return best hypotheses even if unsatisfied.
            if depth == self.max_depth - 1 {
                return DeepeningResult {
                    hypotheses: hyps,
                    depth_used: depth,
                    satisfied:  false,
                };
            }

            log::debug!("Deepening pass {} found no satisfying hypotheses for '{}'", depth + 1, query);
        }

        DeepeningResult { hypotheses: vec![], depth_used: self.max_depth, satisfied: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::ArgNode;
    use crate::types::{ModalType, ModalMode, TypeCategory};
    use petgraph::stable_graph::StableGraph;

    fn node_with_surface(id: u64, surface: &str, score: f32) -> ArgNode {
        let mut n = ArgNode::new(id, NodeType::Concept,
            ModalType::functor(ModalMode::Diamond, TypeCategory::Scene, 1, true), (0,0));
        n.surface = Some(surface.as_bytes().to_vec());
        n.attribution_score = score;
        n.atms_label = 0b1;
        n
    }

    #[test]
    fn finds_satisfying_at_depth_0() {
        let mut g: ArgGraph = StableGraph::new();
        g.add_node(node_with_surface(1, "run", 0.9));
        let sem        = MtlgSemantics::new();
        let perf       = PerfRegistry::new(0.25);
        let mut thresh = ThresholdRegistry::default();
        let deepener   = ProgressiveDeepener::new(None);

        let result = deepener.run(&g, &sem, &perf, &mut thresh, "run");
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

        let result = deepener.run(&g, &sem, &perf, &mut thresh, "nonexistent_predicate");
        assert!(!result.satisfied);
    }
}
