//! Hypothesis generation: ranked candidates from MTLG derivations.
//! TypeCategory::DEFAULT acts as a wildcard in filter_satisfying.

use std::collections::HashMap;
use crate::types::{NodeId, TRDId, ModalType, ModalMode, TypeCategory, Env, Direction};
use crate::arg::{ArgGraph, ArgNode, NodeClass, search::ArgSearch};
use crate::semantics::mtlg_semantics::{MtlgSemantics, LambdaTerm, PropositionGraph};
use crate::adaptive::PerfRegistry;

pub const N_HYPOTHESES: usize = 5;

#[derive(Debug, Clone)]
pub struct Hypothesis {
    pub id:          usize,
    pub lambda_str:  String,
    pub proposition: PropositionGraph,
    pub relevance:   f32,
    pub root_node:   NodeId,
}

impl Hypothesis {
    pub fn from_node(id: usize, node: &ArgNode, semantics: &MtlgSemantics, relevance: f32) -> Self {
        let surface = node.surface_str().unwrap_or("_");
        let term = LambdaTerm::Pred(
            surface.to_string(),
            (0..node.mtlg_type.arity).map(|i| LambdaTerm::Var(format!("x{}", i))).collect(),
        );
        let proposition = semantics.sentence_level(term.clone());
        Self { id, lambda_str: term.display(), proposition, relevance, root_node: node.id }
    }
}

fn node_relevance(node: &ArgNode, active_trd: Option<TRDId>, perf: &PerfRegistry) -> f32 {
    let trd_match = active_trd.map_or(0.5, |trd| {
        if node.trd_membership.contains(&trd) { perf.p_ema(trd) as f32 } else { 0.3 }
    });
    node.attribution_score * trd_match * (1.0 / (1.0 + node.depth))
}

pub fn generate_hypotheses(
    graph:      &ArgGraph,
    semantics:  &MtlgSemantics,
    active_trd: Option<TRDId>,
    perf:       &PerfRegistry,
) -> Vec<Hypothesis> {
    let mut ranked: Vec<_> = graph.node_indices()
        .map(|idx| { let n = &graph[idx]; (n, node_relevance(n, active_trd, perf)) })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.into_iter().take(N_HYPOTHESES).enumerate()
        .map(|(id, (node, rel))| Hypothesis::from_node(id, node, semantics, rel))
        .collect()
}

/// TypeCategory::DEFAULT (0) acts as a wildcard: matches any expected category.
/// Mode must always match exactly.
pub fn filter_satisfying(
    hyps:          Vec<Hypothesis>,
    expected_type: &ModalType,
    type_map:      &HashMap<String, ModalType>,
) -> Vec<Hypothesis> {
    hyps.into_iter().filter(|h| {
        let root_type = type_map.get(&h.proposition.root).copied()
            .unwrap_or(ModalType::atom(ModalMode::Diamond, TypeCategory::DEFAULT));
        let cat_match = expected_type.category == TypeCategory::DEFAULT
            || root_type.category == TypeCategory::DEFAULT
            || root_type.category == expected_type.category;
        root_type.mode == expected_type.mode && cat_match
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeClass};
    use crate::types::{ModalType, ModalMode, TypeCategory, Direction};
    use petgraph::stable_graph::StableGraph;
    use crate::arg::search::ArgGraph;

    fn node(id: u64, surface: &str, score: f32) -> ArgNode {
        let mut n = ArgNode::new(id, NodeClass::DEFAULT,
            ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right), (0,0));
        n.surface = Some(surface.as_bytes().to_vec());
        n.attribution_score = score;
        n.atms_label = 0b1;
        n
    }

    #[test]
    fn generates_by_attribution() {
        let mut g: ArgGraph = StableGraph::new();
        g.add_node(node(1, "run", 0.9));
        g.add_node(node(2, "eat", 0.7));
        let sem = MtlgSemantics::new();
        let perf = PerfRegistry::new(0.25);
        let hyps = generate_hypotheses(&g, &sem, None, &perf);
        assert!(!hyps.is_empty());
        assert_eq!(hyps[0].proposition.root, "run");
    }

    #[test]
    fn default_category_is_wildcard() {
        let mut g: ArgGraph = StableGraph::new();
        g.add_node(node(1, "run", 0.9));
        let sem  = MtlgSemantics::new();
        let perf = PerfRegistry::new(0.25);
        let hyps = generate_hypotheses(&g, &sem, None, &perf);
        let mut type_map = HashMap::new();
        type_map.insert("run".into(),
            ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right));
        let expected = ModalType::atom(ModalMode::Diamond, TypeCategory(5));
        let satisfying = filter_satisfying(hyps, &expected, &type_map);
        assert!(!satisfying.is_empty(), "DEFAULT should match any category");
    }

    #[test]
    fn mode_mismatch_rejected() {
        let mut g: ArgGraph = StableGraph::new();
        g.add_node(node(1, "run", 0.9));
        let sem  = MtlgSemantics::new();
        let perf = PerfRegistry::new(0.25);
        let hyps = generate_hypotheses(&g, &sem, None, &perf);
        let mut type_map = HashMap::new();
        type_map.insert("run".into(),
            ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right));
        let expected = ModalType::atom(ModalMode::Box, TypeCategory::DEFAULT);
        let satisfying = filter_satisfying(hyps, &expected, &type_map);
        assert!(satisfying.is_empty());
    }
}
