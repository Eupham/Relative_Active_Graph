//! Progressive deepening: if hypotheses don't satisfy the query, deepen up to MAX_DEPTH times.
//! Each deepening round expands the ARG one step further and re-generates hypotheses.

use crate::adaptive::{PerfRegistry, ThresholdRegistry};
use crate::arg::{ArgEdge, ArgGraph, ArgNode, EdgeClass, NodeClass};
use crate::generation::{
    hypothesis::{filter_satisfying, generate_hypotheses, Hypothesis},
    traverser::ArgTraverser,
};
use crate::semantics::mtlg_semantics::MtlgSemantics;
use crate::types::{Direction, ModalMode, ModalType, NodeId, TRDId, TypeCategory};

pub const MAX_DEPTH: usize = 3;

/// Result of one deepening round.
#[derive(Debug)]
pub struct DeepeningResult {
    pub hypotheses: Vec<Hypothesis>,
    pub depth_used: usize,
    pub satisfied: bool,
}

/// Progressive deepening orchestrator.
pub struct ProgressiveDeepener {
    pub max_depth: usize,
    pub active_trd: Option<TRDId>,
}

impl ProgressiveDeepener {
    pub fn new(active_trd: Option<TRDId>) -> Self {
        Self {
            max_depth: MAX_DEPTH,
            active_trd,
        }
    }

    /// IDA*-inspired bounded deepening.
    pub fn run(
        &self,
        graph: &ArgGraph,
        semantics: &MtlgSemantics,
        perf: &PerfRegistry,
        thresholds: &mut ThresholdRegistry,
        query_text: &str,
        expected_type: &ModalType,
    ) -> DeepeningResult {
        let budget_per_depth = 50;

        for depth in 0..self.max_depth {
            let mut traverser = ArgTraverser::new(budget_per_depth * (depth + 1), self.active_trd);
            traverser.traverse(graph, perf, thresholds);

            let hyps = generate_hypotheses(graph, semantics, self.active_trd, perf);
            let satisfying = filter_satisfying(hyps.clone(), expected_type, &semantics.type_map);

            if !satisfying.is_empty() {
                return DeepeningResult {
                    hypotheses: satisfying,
                    depth_used: depth,
                    satisfied: true,
                };
            }

            if depth == self.max_depth - 1 {
                return DeepeningResult {
                    hypotheses: hyps,
                    depth_used: depth,
                    satisfied: false,
                };
            }

            log::debug!(
                "Deepening pass {} found no type-satisfying hypotheses",
                depth + 1
            );
        }

        DeepeningResult {
            hypotheses: vec![],
            depth_used: self.max_depth,
            satisfied: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::search::ArgGraph;
    use crate::arg::ArgNode;
    use crate::types::{Direction, ModalMode, ModalType, TypeCategory};
    use petgraph::stable_graph::StableGraph;

    fn node_with_surface(id: u64, surface: &str, score: f32) -> ArgNode {
        let mut n = ArgNode::new(
            id,
            NodeClass::DEFAULT,
            ModalType::functor(
                ModalMode::Diamond,
                TypeCategory::DEFAULT,
                1,
                Direction::Right,
            ),
            (0, 0),
        );
        n.surface = Some(surface.as_bytes().to_vec());
        n.attribution_score = score;
        n.atms_label = 0b1;
        n
    }

    #[test]
    fn finds_satisfying_at_depth_0() {
        let mut g: ArgGraph = StableGraph::new();
        g.add_node(node_with_surface(1, "run", 0.9));
        let sem = MtlgSemantics::new();
        let perf = PerfRegistry::new(0.25);
        let mut thresh = ThresholdRegistry::default();
        let deepener = ProgressiveDeepener::new(None);

        // The node has Diamond/Scene type; register it in the semantics type_map
        // by using a semantics with "run" → Diamond/Scene.
        let mut sem2 = MtlgSemantics::new();
        sem2.register_type(
            "run".into(),
            ModalType::functor(
                ModalMode::Diamond,
                TypeCategory::DEFAULT,
                1,
                Direction::Right,
            ),
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
        let sem = MtlgSemantics::new();
        let perf = PerfRegistry::new(0.25);
        let mut thresh = ThresholdRegistry::default();
        let deepener = ProgressiveDeepener::new(None);

        // Query expects Box/Process — node is Diamond/Scene — no match.
        let expected = ModalType::atom(ModalMode::Box, TypeCategory(1));
        let result = deepener.run(
            &g,
            &sem,
            &perf,
            &mut thresh,
            "nonexistent_predicate",
            &expected,
        );
        assert!(!result.satisfied);
    }
}
