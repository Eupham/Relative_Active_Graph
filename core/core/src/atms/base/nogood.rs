//! Nogood table: set of minimally inconsistent environments.
//! An environment E is inconsistent if any nogood ng satisfies ng ⊆ E.

use smallvec::SmallVec;
use crate::types::Env;
use super::env::subsumes;

/// Stores minimal nogoods (no entry subsumes another).
#[derive(Clone, Debug, Default)]
pub struct NogoodTable {
    nogoods: SmallVec<[Env; 16]>,
}

impl NogoodTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new nogood, maintaining minimality:
    /// - If `env` is subsumed by an existing nogood, skip (existing is more general).
    /// - Remove any existing nogoods subsumed by `env` (env is more general).
    pub fn add(&mut self, env: Env) {
        // If an existing nogood is already at least as general, skip.
        if self.nogoods.iter().any(|&ng| subsumes(ng, env)) {
            return;
        }
        // Remove less-general nogoods (ones that env subsumes).
        self.nogoods.retain(|ng| !subsumes(env, *ng));
        self.nogoods.push(env);
    }

    /// True if `env` contains any minimal inconsistent environment.
    pub fn is_inconsistent(&self, env: Env) -> bool {
        self.nogoods.iter().any(|&ng| subsumes(ng, env))
    }

    pub fn len(&self) -> usize {
        self.nogoods.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nogoods.is_empty()
    }

    pub fn nogoods(&self) -> &[Env] {
        &self.nogoods
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atms::base::env::singleton;

    #[test]
    fn add_and_detect() {
        let mut t = NogoodTable::new();
        let ng = singleton(0) | singleton(1);
        t.add(ng);
        // superset is inconsistent
        assert!(t.is_inconsistent(ng | singleton(2)));
        // non-superset is ok
        assert!(!t.is_inconsistent(singleton(0)));
    }

    #[test]
    fn minimality_maintained() {
        let mut t = NogoodTable::new();
        let general = singleton(0);
        let specific = singleton(0) | singleton(1);
        t.add(specific);
        t.add(general); // more general; specific should be removed
        assert_eq!(t.len(), 1);
        assert_eq!(t.nogoods()[0], general);
    }

    #[test]
    fn subsumed_nogood_skipped() {
        let mut t = NogoodTable::new();
        let general = singleton(0);
        t.add(general);
        t.add(general | singleton(1)); // more specific — skip
        assert_eq!(t.len(), 1);
    }
}
