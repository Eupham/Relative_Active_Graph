//! L-System graph expansion for generative decoding.
//!
//! Treats the seed ArgNode as an Axiom. Applies MetaGrammarEngine rewriting rules
//! iteratively until all branches reach terminal nodes (leaves with no applicable rule).
//!
//! An L-System with zero rules is the identity: every node expands to itself.
//! This is the correct behavior at initialization before rules are inducted.
//! As MetaGrammarEngine discovers rules, expansions grow geometrically.
//!
//! Termination is guaranteed by:
//!   1. Maximum expansion depth `max_depth` (default 32).
//!   2. ATMS label consistency: expansions producing label sets inconsistent with
//!      the active environment are pruned before they recurse.
//!
//! Reference: Prusinkiewicz & Lindenmayer (1990), "The Algorithmic Beauty of Plants",
//! Springer-Verlag. Chapter 1 (DOL-systems and context-free L-systems).

use std::collections::HashMap;
use crate::types::{NodeId, ModalType, Env};
use crate::arg::ArgNode;
use crate::semantics::meta_grammar::{MetaGrammarEngine, GrammarRule, TypedFact};

pub const DEFAULT_MAX_DEPTH: usize = 32;

/// A token produced by L-System expansion.
#[derive(Debug, Clone)]
pub enum GeneratedToken {
    /// A surface byte sequence (terminal leaf).
    Leaf(Vec<u8>),
    /// A node ID referencing an existing ARG node (semi-terminal).
    Node(NodeId),
}

impl GeneratedToken {
    pub fn to_surface(&self, node_map: &HashMap<NodeId, ArgNode>) -> String {
        match self {
            GeneratedToken::Leaf(bytes) => {
                String::from_utf8_lossy(bytes).into_owned()
            }
            GeneratedToken::Node(nid) => {
                node_map.get(nid)
                    .and_then(|n| n.surface_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            }
        }
    }
}

pub struct LSystemExpander<'a> {
    pub meta_grammar: &'a MetaGrammarEngine,
    pub max_depth:    usize,
}

impl<'a> LSystemExpander<'a> {
    pub fn new(meta_grammar: &'a MetaGrammarEngine) -> Self {
        Self { meta_grammar, max_depth: DEFAULT_MAX_DEPTH }
    }

    /// Expand `axiom` into a sequence of GeneratedTokens.
    /// When no rules apply to `axiom`: returns [GeneratedToken::Node(axiom.id)].
    /// This is the identity expansion — correct behavior for an empty rule set.
    pub fn expand(
        &self,
        axiom:      &ArgNode,
        active_env: Env,
        depth:      usize,
    ) -> Vec<GeneratedToken> {
        // Prune expansions inconsistent with the active ATMS environment.
        if axiom.atms_label != 0 && (axiom.atms_label & active_env) == 0 {
            return vec![];
        }

        // Depth limit: return the node itself as a leaf.
        if depth >= self.max_depth {
            return self.terminal(axiom);
        }

        let fact = TypedFact {
            category:  axiom.mtlg_type.category,
            mode:      axiom.mtlg_type.mode,
            direction: axiom.mtlg_type.direction,
        };

        // Find the highest-support rule whose first antecedent matches this node's type.
        let best_rule: Option<&GrammarRule> = self.meta_grammar.rules.iter()
            .filter(|r| r.antecedents.first() == Some(&fact))
            .max_by(|a, b| a.support.partial_cmp(&b.support)
                .unwrap_or(std::cmp::Ordering::Equal));

        match best_rule {
            None => {
                // No rule applies. Identity expansion: return node as leaf.
                self.terminal(axiom)
            }
            Some(rule) => {
                // Materialize child nodes for each antecedent fact beyond the first.
                let children: Vec<ArgNode> = rule.antecedents.iter().skip(1)
                    .enumerate()
                    .map(|(i, fact)| self.materialize_child(axiom, fact, i))
                    .collect();

                // Recurse: first expand the axiom itself (it may sub-expand further
                // if its own type has a matching rule after producing the consequent),
                // then expand each child.
                let mut result = self.expand_as_consequent(axiom, rule, active_env, depth);
                for child in &children {
                    result.extend(self.expand(child, active_env, depth + 1));
                }
                result
            }
        }
    }

    fn terminal(&self, node: &ArgNode) -> Vec<GeneratedToken> {
        match &node.surface {
            Some(bytes) if !bytes.is_empty() => vec![GeneratedToken::Leaf(bytes.clone())],
            _ => vec![GeneratedToken::Node(node.id)],
        }
    }

    fn materialize_child(
        &self,
        parent: &ArgNode,
        fact:   &TypedFact,
        index:  usize,
    ) -> ArgNode {
        let mut child = parent.clone();
        // Assign a deterministic child ID derived from parent + index.
        let combined = parent.id.wrapping_add(index as u64).wrapping_mul(0x9e3779b97f4a7c15);
        child.id = combined;
        child.mtlg_type = ModalType {
            category:  fact.category,
            mode:      fact.mode,
            direction: fact.direction,
            arity:     0,
        };
        child.atms_label = parent.atms_label;
        child.surface    = None;  // no surface until the L-System derives one
        child
    }

    fn expand_as_consequent(
        &self,
        axiom:      &ArgNode,
        rule:       &GrammarRule,
        active_env: Env,
        depth:      usize,
    ) -> Vec<GeneratedToken> {
        // The axiom takes on the consequent type and attempts further expansion.
        let mut consequent_node = axiom.clone();
        consequent_node.mtlg_type = ModalType {
            category:  rule.consequent.category,
            mode:      rule.consequent.mode,
            direction: rule.consequent.direction,
            arity:     0,
        };
        // If the consequent type differs from the axiom type, recurse.
        // Otherwise return the terminal to avoid infinite loops.
        if consequent_node.mtlg_type.category != axiom.mtlg_type.category
            || consequent_node.mtlg_type.mode != axiom.mtlg_type.mode
        {
            self.expand(&consequent_node, active_env, depth + 1)
        } else {
            self.terminal(axiom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantics::meta_grammar::MetaGrammarEngine;
    use crate::types::{ModalType, ModalMode, TypeCategory, Direction};
    use crate::arg::{ArgNode, NodeClass};

    fn make_node(id: u64) -> ArgNode {
        ArgNode::new(id, NodeClass::DEFAULT, ModalType::default(), (0, 0))
    }

    fn make_node_with_surface(id: u64, surface: &str) -> ArgNode {
        let mut n = make_node(id);
        n.surface = Some(surface.as_bytes().to_vec());
        n
    }

    #[test]
    fn empty_rules_identity_expansion() {
        let engine = MetaGrammarEngine::new(0.0);
        let expander = LSystemExpander::new(&engine);
        let node = make_node_with_surface(1, "hello");
        let tokens = expander.expand(&node, u64::MAX, 0);
        assert_eq!(tokens.len(), 1);
        match &tokens[0] {
            GeneratedToken::Leaf(b) => assert_eq!(b, b"hello"),
            _ => panic!("expected leaf"),
        }
    }

    #[test]
    fn env_mismatch_prunes() {
        let engine = MetaGrammarEngine::new(0.0);
        let expander = LSystemExpander::new(&engine);
        let mut node = make_node(1);
        node.atms_label = 0b10; // bit 1
        let tokens = expander.expand(&node, 0b01, 0); // active_env has bit 0
        assert!(tokens.is_empty());
    }
}
