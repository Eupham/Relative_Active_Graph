//! Horn-clause ATMS node.
//! Base layer invariant: LABEL(v) = singleton {Env} — one minimal environment.
//! Propagation is O(1) per justification because no label-set management is needed.

use crate::types::{NodeId, Env};
use super::env::subsumes;

/// Whether a node is an assumption (primitive) or derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Assumption,
    Derived,
}

/// An ATMS node in the base (Horn-clause) layer.
#[derive(Clone, Debug)]
pub struct BaseNode {
    pub id:   NodeId,
    pub kind: NodeKind,
    /// The singleton label: None = not yet derived.
    /// For assumptions, set at construction time to their singleton bit env.
    pub label: Option<Env>,
}

impl BaseNode {
    /// Create an assumption node with its own singleton environment bit.
    pub fn assumption(id: NodeId, env_bit: u8) -> Self {
        let env = 1u64 << env_bit;
        Self { id, kind: NodeKind::Assumption, label: Some(env) }
    }

    /// Create a derived node with no label yet.
    pub fn derived(id: NodeId) -> Self {
        Self { id, kind: NodeKind::Derived, label: None }
    }

    /// The node is derived in (supported by) `active_env` if its label is a subset of it.
    pub fn is_active_in(&self, active_env: Env) -> bool {
        self.label.map_or(false, |l| subsumes(l, active_env))
    }

    /// Update label: returns true if the label changed (triggers propagation).
    /// For Horn-clause base layer: the label can only be set once (monotone).
    pub fn set_label(&mut self, env: Env) -> bool {
        match self.label {
            None => {
                self.label = Some(env);
                true
            }
            Some(existing) if existing != env => {
                // In Horn base layer, this means multiple derivation paths exist.
                // Take the union (least upper bound) — safe because we only ever add
                // assumptions, never retract them in the base layer.
                self.label = Some(existing | env);
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atms::base::env::singleton;

    #[test]
    fn assumption_active_in_superset() {
        let n = BaseNode::assumption(1, 3);
        assert!(n.is_active_in(singleton(3) | singleton(5)));
    }

    #[test]
    fn assumption_inactive_without_bit() {
        let n = BaseNode::assumption(1, 3);
        assert!(!n.is_active_in(singleton(5)));
    }

    #[test]
    fn derived_inactive_until_labeled() {
        let n = BaseNode::derived(2);
        assert!(!n.is_active_in(singleton(0) | singleton(1)));
    }

    #[test]
    fn set_label_monotone() {
        let mut n = BaseNode::derived(3);
        assert!(n.set_label(singleton(0)));
        assert!(!n.set_label(singleton(0))); // no change
        assert!(n.set_label(singleton(1)));  // union: new bits added
        assert_eq!(n.label, Some(singleton(0) | singleton(1)));
    }
}
