//! Horn-clause ATMS node.
//! Base layer invariant: LABEL(v) = one minimal environment.
//! A second derivation is accepted only if it is strictly cheaper (fewer assumptions).
//! Multi-justification label sets are handled by the BF-ATMS layer.

use crate::types::{NodeId, Env};
use super::env::subsumes;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Assumption,
    Derived,
}

#[derive(Clone, Debug)]
pub struct BaseNode {
    pub id:   NodeId,
    pub kind: NodeKind,
    pub label: Option<Env>,
}

impl BaseNode {
    pub fn assumption(id: NodeId, env_bit: u8) -> Self {
        let env = 1u64 << env_bit;
        Self { id, kind: NodeKind::Assumption, label: Some(env) }
    }

    pub fn derived(id: NodeId) -> Self {
        Self { id, kind: NodeKind::Derived, label: None }
    }

    pub fn is_active_in(&self, active_env: Env) -> bool {
        self.label.map_or(false, |l| subsumes(l, active_env))
    }

    /// Update label under Horn-clause monotonicity:
    ///
    /// - If unlabelled: accept unconditionally.
    /// - If the new env is a strict subset of the existing label: it represents a
    ///   cheaper proof (fewer assumptions required); replace.
    /// - Otherwise: discard. The existing label is already minimal or equally
    ///   minimal. Unioning would incorrectly conjoin independent proofs.
    ///
    /// Returns true only when the stored label actually changes, so propagation
    /// queues are triggered only on genuine updates.
    pub fn set_label(&mut self, env: Env) -> bool {
        match self.label {
            None => {
                self.label = Some(env);
                true
            }
            Some(existing) => {
                // Accept iff new env is a proper subset: subsumes(env, existing)
                // means every bit of env is in existing, i.e. env ⊆ existing.
                // We additionally require env != existing so it is strictly cheaper.
                if subsumes(env, existing) && env != existing {
                    self.label = Some(env);
                    true
                } else {
                    false
                }
            }
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
    fn set_label_first_time_always_accepted() {
        let mut n = BaseNode::derived(3);
        assert!(n.set_label(singleton(0) | singleton(1)));
        assert_eq!(n.label, Some(singleton(0) | singleton(1)));
    }

    #[test]
    fn set_label_same_env_rejected() {
        let mut n = BaseNode::derived(3);
        n.set_label(singleton(0));
        assert!(!n.set_label(singleton(0)), "identical label must not trigger propagation");
    }

    #[test]
    fn set_label_superset_rejected() {
        let mut n = BaseNode::derived(3);
        n.set_label(singleton(0));
        // singleton(0) | singleton(1) is a *superset* — more expensive, must be rejected
        assert!(!n.set_label(singleton(0) | singleton(1)));
        assert_eq!(n.label, Some(singleton(0)));
    }

    #[test]
    fn set_label_subset_accepted_replaces() {
        let mut n = BaseNode::derived(3);
        n.set_label(singleton(0) | singleton(1));
        // singleton(0) alone is strictly cheaper — should replace
        assert!(n.set_label(singleton(0)));
        assert_eq!(n.label, Some(singleton(0)));
    }
}
