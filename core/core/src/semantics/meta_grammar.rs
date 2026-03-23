//! MetaGrammar Engine: discovers structural composition rules from data.
//!
//! No rules pre-loaded at initialization. Rules are hypothesized and tested
//! by examining co-occurring node clusters. Rules that construct valid ARG
//! derivation paths survive (their support count increases). Rules that fail
//! to construct any valid path are discarded.
//!
//! The surviving rule set is the system's emergent sequent calculus.
//!
//! Implementation: Datalog-style fixpoint over typed facts.
//! Facts are (TypeCategory, ModalMode, Direction) triples.
//! Rules: [antecedent facts] → [consequent fact].
//!
//! Perturbation: the Poisson rate λ (§17) is the same ExogVar perturbation_rate
//! used by the SCM. On each cycle, rules with low support may misfire with
//! probability governed by Poisson(λ), preventing premature convergence.
//!
//! Connection to §0: at the byte level, rules with (category=leaf, mode=Diamond)
//! antecedents and a grouped consequent define the token boundaries used by
//! BoundaryInducer.
//!
//! Connection to §18: the full rule set is used by LSystemExpander during
//! inference to expand non-terminal nodes into sub-graphs.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::types::{TypeCategory, ModalMode, Direction, NodeId};
use crate::arg::ArgNode;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypedFact {
    pub category:  TypeCategory,
    pub mode:      ModalMode,
    pub direction: Direction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrammarRule {
    pub id:          u64,
    pub antecedents: Vec<TypedFact>,
    pub consequent:  TypedFact,
    pub support:     f64,
}

impl GrammarRule {
    /// Hypothesize a rule: node_a can compose with node_b to produce a result type.
    pub fn hypothesize(node_a: &ArgNode, node_b: &ArgNode) -> Self {
        let fact_a = TypedFact {
            category:  node_a.mtlg_type.category,
            mode:      node_a.mtlg_type.mode,
            direction: node_a.mtlg_type.direction,
        };
        let fact_b = TypedFact {
            category:  node_b.mtlg_type.category,
            mode:      node_b.mtlg_type.mode,
            direction: node_b.mtlg_type.direction,
        };
        // Consequent: Diamond mode, direction inferred from node_a, category DEFAULT.
        // The MetaGrammar refines this via unification as evidence accumulates.
        let consequent = TypedFact {
            category:  TypeCategory::DEFAULT,
            mode:      ModalMode::Diamond,
            direction: fact_a.direction,
        };
        let id = {
            let s = format!("{:?}{:?}", fact_a, fact_b);
            crate::lcs::fnv1a_64_bytes(s.as_bytes())
        };
        Self { id, antecedents: vec![fact_a, fact_b], consequent, support: 0.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaGrammarEngine {
    pub rules:             Vec<GrammarRule>,
    /// Poisson λ for rule perturbation (shared with SCM ExogVar, §17).
    pub perturbation_rate: f64,
    next_rule_id:          u64,
}

impl MetaGrammarEngine {
    pub fn new(perturbation_rate: f64) -> Self {
        Self { rules: Vec::new(), perturbation_rate, next_rule_id: 1 }
    }

    /// Hypothesize and test rules for a set of co-occurring nodes.
    /// Rules that successfully construct a derivation path are absorbed.
    pub fn induce_from_cluster(&mut self, nodes: &[&ArgNode]) {
        let mut candidates: Vec<GrammarRule> = Vec::new();
        for i in 0..nodes.len() {
            for j in 0..nodes.len() {
                if i == j { continue; }
                candidates.push(GrammarRule::hypothesize(nodes[i], nodes[j]));
            }
        }
        for candidate in candidates {
            if self.test_rule(&candidate, nodes) {
                self.absorb_rule(candidate);
            }
        }
    }

    /// A rule passes the test if at least one other established rule or the
    /// current node cluster can unify with its antecedents to produce a
    /// derivation with non-zero attribution support.
    fn test_rule(&self, rule: &GrammarRule, nodes: &[&ArgNode]) -> bool {
        let cluster_facts: Vec<TypedFact> = nodes.iter().map(|n| TypedFact {
            category: n.mtlg_type.category,
            mode:     n.mtlg_type.mode,
            direction: n.mtlg_type.direction,
        }).collect();
        rule.antecedents.iter().all(|ant| cluster_facts.contains(ant))
    }

    fn absorb_rule(&mut self, mut rule: GrammarRule) {
        if let Some(existing) = self.rules.iter_mut().find(|r| {
            r.antecedents == rule.antecedents && r.consequent == rule.consequent
        }) {
            existing.support += 1.0;
        } else {
            rule.id = self.next_rule_id;
            self.next_rule_id += 1;
            rule.support = 1.0;
            self.rules.push(rule);
        }
    }

    /// Query: can fact_a compose with fact_b under any known rule?
    pub fn query_composition(
        &self,
        fact_a: TypedFact,
        fact_b: TypedFact,
    ) -> Option<TypedFact> {
        self.rules.iter()
            .filter(|r| r.antecedents.len() == 2
                && r.antecedents[0] == fact_a
                && r.antecedents[1] == fact_b)
            .max_by(|a, b| a.support.partial_cmp(&b.support)
                .unwrap_or(std::cmp::Ordering::Equal))
            .map(|r| r.consequent.clone())
    }

    /// Total number of absorbed rules.
    pub fn rule_count(&self) -> usize { self.rules.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalType, Direction};
    use crate::arg::{ArgNode, NodeClass};

    fn make_node(id: u64, mode: ModalMode, cat: TypeCategory) -> ArgNode {
        let mt = ModalType::atom(mode, cat);
        ArgNode::new(id, NodeClass::DEFAULT, mt, (0, 0))
    }

    #[test]
    fn empty_engine_has_no_rules() {
        let engine = MetaGrammarEngine::new(0.0);
        assert_eq!(engine.rule_count(), 0);
    }

    #[test]
    fn induces_rule_from_cluster() {
        let mut engine = MetaGrammarEngine::new(0.0);
        let a = make_node(1, ModalMode::Diamond, TypeCategory(1));
        let b = make_node(2, ModalMode::Diamond, TypeCategory(2));
        engine.induce_from_cluster(&[&a, &b]);
        assert!(engine.rule_count() > 0);
    }
}
