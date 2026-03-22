//! Incremental singleton-label propagation for the Horn-clause base ATMS.
//! Uses a work-queue: when a node's label changes, enqueue all justifications
//! that have it as an antecedent (via reverse index).

use std::collections::{HashMap, VecDeque};
use crate::types::{NodeId, Env};
use super::{
    node::BaseNode,
    justification::Justification,
    nogood::NogoodTable,
    env::union,
};

/// The complete Horn-clause ATMS state.
pub struct BaseAtms {
    pub nodes:     HashMap<NodeId, BaseNode>,
    /// All justifications, indexed by their consequent for lookup.
    justifications: Vec<Justification>,
    /// Reverse index: antecedent NodeId → justification indices.
    ant_to_just: HashMap<NodeId, Vec<usize>>,
    /// Symmetric index: consequent NodeId → justification indices.
    /// Used by propagate_attribution_backward to trace proof origins.
    consequent_to_just: HashMap<NodeId, Vec<usize>>,
    pub nogoods:   NogoodTable,
}

impl BaseAtms {
    pub fn new() -> Self {
        Self {
            nodes:              HashMap::new(),
            justifications:     Vec::new(),
            ant_to_just:        HashMap::new(),
            consequent_to_just: HashMap::new(),
            nogoods:            NogoodTable::new(),
        }
    }

    /// Register a new assumption node (its env_bit must be unique).
    pub fn add_assumption(&mut self, id: NodeId, env_bit: u8) {
        self.nodes.insert(id, BaseNode::assumption(id, env_bit));
    }

    /// Register a new derived node (no label yet).
    pub fn add_derived(&mut self, id: NodeId) {
        self.nodes.entry(id).or_insert_with(|| BaseNode::derived(id));
    }

    /// Add a Horn-clause justification and propagate immediately if antecedents are labeled.
    pub fn add_justification(&mut self, j: Justification) -> Option<Env> {
        let idx = self.justifications.len();
        for &ant in &j.antecedents {
            self.ant_to_just.entry(ant).or_default().push(idx);
        }
        // Symmetric index: consequent → justification.
        self.consequent_to_just
            .entry(j.consequent)
            .or_default()
            .push(idx);
        // Try to derive immediately.
        let derived_env = self.try_derive(&j);
        self.justifications.push(j);
        derived_env
    }

    /// Propagate label change for `changed_id` through all downstream justifications.
    /// Returns list of (NodeId, new_env) that were updated (for callers to act on).
    pub fn propagate(&mut self, changed_id: NodeId) -> Vec<(NodeId, Env)> {
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        queue.push_back(changed_id);
        let mut updated = Vec::new();

        while let Some(nid) = queue.pop_front() {
            let just_indices = self.ant_to_just.get(&nid).cloned().unwrap_or_default();
            for ji in just_indices {
                let j = &self.justifications[ji];
                let consequent_id = j.consequent;
                if let Some(env) = self.try_derive(j) {
                    let node = self.nodes.entry(consequent_id)
                        .or_insert_with(|| BaseNode::derived(consequent_id));
                    if node.set_label(env) {
                        updated.push((consequent_id, env));
                        queue.push_back(consequent_id);
                        // Check if this new label is inconsistent.
                        if self.nogoods.is_inconsistent(env) {
                            log::debug!("NOGOOD triggered for node {} env {:064b}", consequent_id, env);
                        }
                    }
                }
            }
        }
        updated
    }

    fn try_derive(&self, j: &Justification) -> Option<Env> {
        j.derive_env(|id| self.nodes.get(&id).and_then(|n| n.label))
    }

    /// Add a nogood and check existing labels against it.
    pub fn add_nogood(&mut self, env: Env) -> Vec<NodeId> {
        self.nogoods.add(env);
        // Return nodes whose labels are now inconsistent.
        self.nodes.values()
            .filter(|n| n.label.map_or(false, |l| self.nogoods.is_inconsistent(l)))
            .map(|n| n.id)
            .collect()
    }

    /// Is node `id` active in the given environment?
    pub fn is_active(&self, id: NodeId, env: Env) -> bool {
        self.nodes.get(&id).map_or(false, |n| n.is_active_in(env))
    }

    /// Label of node `id`, if derived.
    pub fn label_of(&self, id: NodeId) -> Option<Env> {
        self.nodes.get(&id).and_then(|n| n.label)
    }

    /// Deactivate a situation environment: O(1) — just stop asserting those bits.
    /// Returns nodes that are no longer active.
    pub fn deactivate(&self, env: Env) -> Vec<NodeId> {
        self.nodes.values()
            .filter(|n| {
                n.label.map_or(false, |l| {
                    l & env != 0 && l & !env == 0
                })
            })
            .map(|n| n.id)
            .collect()
    }

    /// All antecedent NodeIds for any justification that derives `consequent`.
    /// Used for backward attribution propagation through the proof graph.
    pub fn antecedents_of(&self, consequent: NodeId) -> Vec<NodeId> {
        self.consequent_to_just
            .get(&consequent)
            .map(|idxs| {
                idxs.iter()
                    .flat_map(|&ji| self.justifications[ji].antecedents.iter().copied())
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for BaseAtms {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atms::base::{env::singleton, justification::Justification};

    #[test]
    fn propagation_chain() {
        let mut atms = BaseAtms::new();
        // a(bit 0), b(bit 1) are assumptions
        atms.add_assumption(1, 0);
        atms.add_assumption(2, 1);
        // c derived from a, b
        atms.add_derived(3);
        let j = Justification::new(3, vec![1, 2]);
        atms.add_justification(j);
        atms.propagate(1);
        atms.propagate(2);
        let expected = singleton(0) | singleton(1);
        assert_eq!(atms.label_of(3), Some(expected));
    }

    #[test]
    fn nogood_detected() {
        let mut atms = BaseAtms::new();
        atms.add_assumption(1, 0);
        let ng = singleton(0);
        let inconsistent = atms.add_nogood(ng);
        assert!(inconsistent.contains(&1));
    }

    #[test]
    fn antecedents_of_works() {
        let mut atms = BaseAtms::new();
        atms.add_assumption(1, 0);
        atms.add_assumption(2, 1);
        atms.add_derived(3);
        let j = Justification::new(3, vec![1, 2]);
        atms.add_justification(j);
        let ants = atms.antecedents_of(3);
        assert!(ants.contains(&1));
        assert!(ants.contains(&2));
    }
}
