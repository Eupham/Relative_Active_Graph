//! Horn-clause justifications for the base ATMS layer.
//! Each justification: (consequent: NodeId, antecedents: [NodeId]).
//! Watch-literal optimization: fires only when a watched antecedent label changes.

use crate::types::{NodeId, Env};
use super::env::union;

/// A Horn-clause justification: one consequent, N antecedents.
#[derive(Clone, Debug)]
pub struct Justification {
    pub consequent:  NodeId,
    pub antecedents: Vec<NodeId>,
    /// The two "watched" antecedents (indices into `antecedents`).
    /// Propagation only re-evaluates when a watched antecedent changes.
    watched: [usize; 2],
}

impl Justification {
    pub fn new(consequent: NodeId, antecedents: Vec<NodeId>) -> Self {
        let w0 = 0;
        let w1 = antecedents.len().saturating_sub(1).min(1);
        Self { consequent, antecedents, watched: [w0, w1] }
    }

    /// Fact justification: no antecedents — consequent is derived from the empty env.
    pub fn fact(consequent: NodeId) -> Self {
        Self { consequent, antecedents: vec![], watched: [0, 0] }
    }

    /// Compute the environment that this justification would derive for its consequent,
    /// given the current labels of all antecedents.
    /// Returns None if any antecedent has no label (not yet derived).
    pub fn derive_env<F>(&self, label_of: F) -> Option<Env>
    where
        F: Fn(NodeId) -> Option<Env>,
    {
        let mut combined = Env::default(); // 0 = empty environment
        for &ant in &self.antecedents {
            let lbl = label_of(ant)?;
            combined = union(combined, lbl);
        }
        Some(combined)
    }

    /// Returns the watched antecedent NodeIds (for the propagation queue).
    pub fn watched_antecedents(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.watched
            .iter()
            .filter_map(move |&wi| self.antecedents.get(wi).copied())
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Promote a non-watched antecedent to a watch slot when the first watch is satisfied.
    pub fn advance_watch(&mut self, slot: usize, nodes_len: usize) {
        if nodes_len <= 2 { return; }
        // Move watch to next antecedent not already watched
        let current = self.watched[slot];
        let other   = self.watched[1 - slot];
        let next = (current + 1..nodes_len).find(|&i| i != other);
        if let Some(n) = next {
            self.watched[slot] = n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atms::base::env::singleton;

    #[test]
    fn derive_env_combines_antecedents() {
        let j = Justification::new(99, vec![1, 2, 3]);
        let env = j.derive_env(|id| match id {
            1 => Some(singleton(0)),
            2 => Some(singleton(1)),
            3 => Some(singleton(2)),
            _ => None,
        });
        assert_eq!(env, Some(singleton(0) | singleton(1) | singleton(2)));
    }

    #[test]
    fn derive_env_missing_antecedent() {
        let j = Justification::new(99, vec![1, 2]);
        let env = j.derive_env(|id| if id == 1 { Some(singleton(0)) } else { None });
        assert_eq!(env, None);
    }

    #[test]
    fn fact_derives_empty_env() {
        let j = Justification::fact(42);
        let env = j.derive_env(|_| None); // no antecedents needed
        assert_eq!(env, Some(0u64));
    }
}
