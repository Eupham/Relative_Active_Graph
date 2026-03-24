//! Bridge: MetaGrammarEngine → BoundaryInducer (GAP 3).
//!
//! Exports grammar rules as serializable `CfgRule` objects that the Python
//! `BoundaryInducer` can consume via IPC (JSON over stdin/stdout or HTTP).
//!
//! The bridge filters rules whose antecedents are leaf-mode Diamond facts
//! (character-level composition) and translates them into `CfgRule` objects
//! that define token boundaries. This closes the feedback loop:
//!
//!   characters → MetaGrammarEngine rules → CfgRule export → BoundaryInducer
//!   → updated tokenization → new character graph → new rules → …
//!
//! Usage from server.py (or any IPC host):
//!   1. Call `export_boundary_rules(&engine)` to get `Vec<CfgRule>`.
//!   2. Serialize to JSON via serde.
//!   3. Send to Python BoundaryInducer.register_rule(cfg_rule_dict).

use serde::{Serialize, Deserialize};
use crate::types::{TypeCategory, ModalMode};
use crate::semantics::{MetaGrammarEngine, GrammarRule, TypedFact};

/// A context-free grammar rule in the format expected by BoundaryInducer.
///
/// Maps to Python `CfgRule(lhs, rhs, boundary_type, confidence)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CfgRule {
    /// Left-hand side category (the grouped/composed result).
    pub lhs: u32,
    /// Right-hand side categories (the leaf/character-level constituents).
    pub rhs: Vec<u32>,
    /// Boundary type: "merge" (characters compose into token) or
    /// "split" (token boundary detected between characters).
    pub boundary_type: BoundaryType,
    /// Confidence from the grammar rule's support count, normalized to [0,1].
    pub confidence: f64,
    /// Source rule ID for provenance tracking.
    pub source_rule_id: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BoundaryType {
    /// Characters should merge into a single token.
    Merge,
    /// A token boundary should be placed here.
    Split,
}

/// Returns true if a TypedFact represents a leaf-level (character/byte) fact.
fn is_leaf_fact(fact: &TypedFact) -> bool {
    fact.mode == ModalMode::Diamond && fact.category.0 <= 2
}

/// Returns true if a GrammarRule is a boundary-defining rule.
fn is_boundary_rule(rule: &GrammarRule) -> bool {
    rule.antecedents.len() >= 2
        && rule.antecedents.iter().all(|a| is_leaf_fact(a))
        && rule.consequent.category.0 > 0
}

fn infer_boundary_type(rule: &GrammarRule) -> BoundaryType {
    match rule.consequent.mode {
        ModalMode::Diamond => BoundaryType::Merge,
        _ => BoundaryType::Split,
    }
}

fn normalize_support(support: f64, max_support: f64) -> f64 {
    if max_support <= 0.0 { return 0.0; }
    (support / max_support).min(1.0)
}

/// Export boundary-relevant grammar rules as `CfgRule` objects.
pub fn export_boundary_rules(engine: &MetaGrammarEngine) -> Vec<CfgRule> {
    let max_support = engine.rules.iter()
        .map(|r| r.support)
        .fold(0.0f64, f64::max);

    engine.rules.iter()
        .filter(|r| is_boundary_rule(r))
        .map(|r| CfgRule {
            lhs: r.consequent.category.0,
            rhs: r.antecedents.iter().map(|a| a.category.0).collect(),
            boundary_type: infer_boundary_type(r),
            confidence: normalize_support(r.support, max_support),
            source_rule_id: r.id,
        })
        .collect()
}

/// Export ALL grammar rules as CfgRules.
pub fn export_all_rules(engine: &MetaGrammarEngine) -> Vec<CfgRule> {
    let max_support = engine.rules.iter()
        .map(|r| r.support)
        .fold(0.0f64, f64::max);

    engine.rules.iter()
        .map(|r| CfgRule {
            lhs: r.consequent.category.0,
            rhs: r.antecedents.iter().map(|a| a.category.0).collect(),
            boundary_type: infer_boundary_type(r),
            confidence: normalize_support(r.support, max_support),
            source_rule_id: r.id,
        })
        .collect()
}

/// Serialize rules to JSON for IPC transport.
pub fn rules_to_json(rules: &[CfgRule]) -> String {
    serde_json::to_string(rules).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalType, Direction};
    use crate::arg::{ArgNode, NodeClass};

    fn make_leaf_node(id: u64, cat: u32) -> ArgNode {
        let mt = ModalType::atom(ModalMode::Diamond, TypeCategory(cat));
        ArgNode::new(id, NodeClass::DEFAULT, mt, (0, 0))
    }

    fn make_box_node(id: u64, cat: u32) -> ArgNode {
        let mt = ModalType::atom(ModalMode::Box, TypeCategory(cat));
        ArgNode::new(id, NodeClass::DEFAULT, mt, (0, 0))
    }

    #[test]
    fn exports_boundary_rules_from_leaf_cluster() {
        let mut engine = MetaGrammarEngine::new(0.0);
        let a = make_leaf_node(1, 1);
        let b = make_leaf_node(2, 2);
        engine.induce_from_cluster(&[&a, &b]);

        let rules = export_boundary_rules(&engine);
        assert!(!rules.is_empty(), "should export at least one boundary rule");
        for r in &rules {
            assert_eq!(r.boundary_type, BoundaryType::Merge);
            assert!(r.confidence > 0.0);
        }
    }

    #[test]
    fn non_leaf_rules_excluded_from_boundary_export() {
        let mut engine = MetaGrammarEngine::new(0.0);
        let a = make_box_node(1, 5);
        let b = make_box_node(2, 6);
        engine.induce_from_cluster(&[&a, &b]);

        let boundary = export_boundary_rules(&engine);
        assert!(boundary.is_empty());

        let all = export_all_rules(&engine);
        assert!(!all.is_empty());
    }

    #[test]
    fn json_roundtrip() {
        let rules = vec![CfgRule {
            lhs: 3, rhs: vec![1, 2],
            boundary_type: BoundaryType::Merge,
            confidence: 0.8, source_rule_id: 42,
        }];
        let json = rules_to_json(&rules);
        let parsed: Vec<CfgRule> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].lhs, 3);
    }
}
