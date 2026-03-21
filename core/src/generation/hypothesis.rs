//! Hypothesis generation: N candidates from MTLG derivations, ranked by relevance in G(s).

use std::collections::HashMap;
use crate::types::{NodeId, TRDId, ModalType, ModalMode, TypeCategory, Env};
use crate::arg::{ArgGraph, ArgNode, search::ArgSearch};
use crate::semantics::mtlg_semantics::{MtlgSemantics, LambdaTerm, PropositionGraph};
use crate::adaptive::PerfRegistry;
use petgraph::visit::EdgeRef;

/// Maximum hypotheses to generate per query.
pub const N_HYPOTHESES: usize = 5;

/// A single hypothesis: a candidate λ-expression + its ARG provenance.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    pub id:          usize,
    pub lambda_str:  String,
    pub proposition: PropositionGraph,
    pub relevance:   f32,
    pub root_node:   NodeId,
}

impl Hypothesis {
    pub fn from_node(
        id:        usize,
        node:      &ArgNode,
        semantics: &MtlgSemantics,
        relevance: f32,
    ) -> Self {
        // Build a simple predicate λ-term from the node's surface form and type.
        let surface = node.surface_str().unwrap_or("_");
        let term = LambdaTerm::Pred(
            surface.to_string(),
            (0..node.mtlg_type.arity)
                .map(|i| LambdaTerm::Var(format!("x{}", i)))
                .collect(),
        );
        let proposition = semantics.sentence_level(term.clone());
        Self {
            id,
            lambda_str:  term.display(),
            proposition,
            relevance,
            root_node:   node.id,
        }
    }
}

/// Relevance score of a node in G(s) given the active TRD.
fn node_relevance(node: &ArgNode, active_trd: Option<TRDId>, perf: &PerfRegistry) -> f32 {
    let trd_match = active_trd.map_or(0.5, |trd| {
        if node.trd_membership.contains(&trd) {
            perf.p_ema(trd) as f32
        } else {
            0.3
        }
    });
    node.attribution_score * trd_match * (1.0 / (1.0 + node.depth))
}

/// Generate N hypotheses from the ARG, ranked by TRD-relative relevance.
pub fn generate_hypotheses(
    graph:      &ArgGraph,
    semantics:  &MtlgSemantics,
    active_trd: Option<TRDId>,
    perf:       &PerfRegistry,
) -> Vec<Hypothesis> {
    // Rank all nodes by relevance.
    let mut ranked: Vec<_> = graph.node_indices()
        .map(|idx| {
            let node = &graph[idx];
            let rel  = node_relevance(node, active_trd, perf);
            (node, rel)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top N, build hypotheses.
    ranked.into_iter().take(N_HYPOTHESES).enumerate().map(|(id, (node, rel))| {
        Hypothesis::from_node(id, node, semantics, rel)
    }).collect()
}

/// Filter hypotheses that satisfy the query (basic: hypothesis root matches query predicate).
pub fn filter_satisfying(hyps: Vec<Hypothesis>, query_predicate: &str) -> Vec<Hypothesis> {
    hyps.into_iter().filter(|h| {
        h.proposition.root == query_predicate || h.lambda_str.contains(query_predicate)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeType};
    use crate::types::{ModalType, ModalMode, TypeCategory};
    use petgraph::stable_graph::StableGraph;

    fn node(id: u64, surface: &str, score: f32) -> ArgNode {
        let mut n = ArgNode::new(id, NodeType::Concept,
            ModalType::functor(ModalMode::Diamond, TypeCategory::Scene, 1, true), (0,0));
        n.surface = Some(surface.as_bytes().to_vec());
        n.attribution_score = score;
        n.atms_label = 0b1;
        n
    }

    #[test]
    fn generates_hypotheses_from_graph() {
        let mut g: ArgGraph = StableGraph::new();
        g.add_node(node(1, "run", 0.9));
        g.add_node(node(2, "eat", 0.7));
        let sem  = MtlgSemantics::new();
        let perf = PerfRegistry::new(0.25);
        let hyps = generate_hypotheses(&g, &sem, None, &perf);
        assert!(!hyps.is_empty());
        assert_eq!(hyps[0].proposition.root, "run"); // highest attribution first
    }
}
