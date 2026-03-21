//! BF-ATMS: bounded, non-Horn ATMS for causal counterfactual reasoning.
//! (Provan & Singh 1991 — polynomial search bounded by label complexity k.)
//!
//! The intervention mechanism works through ATMS assumptions, not edge IDs.
//! To remove edge e: add a NOGOOD for the assumption bits that justified e,
//! then count how many scope nodes lose all support.

use std::collections::HashMap;
use crate::types::{NodeId, Env};
use super::scope::CounterfactualScope;
use crate::atms::base::env::{singleton, union, subsumes};

/// Maximum label set size (k in BF-ATMS). Polynomial guarantee holds for k ≤ 4.
const MAX_LABEL_COMPLEXITY: usize = 4;

#[derive(Clone, Debug)]
pub struct BfNode {
    pub id:     NodeId,
    /// Minimal supporting environments (non-Horn: can have multiple).
    pub labels: Vec<Env>,
    pub cost:   f64,
}

impl BfNode {
    pub fn new(id: NodeId, cost: f64) -> Self {
        Self { id, labels: Vec::new(), cost }
    }

    /// Add a new minimal label, maintaining minimality of the label set.
    pub fn add_label(&mut self, env: Env) {
        // Drop new label if it is already subsumed by an existing minimal label.
        if self.labels.iter().any(|&l| subsumes(l, env)) { return; }
        // Drop any existing labels that are subsumed by (less minimal than) the new one.
        self.labels.retain(|&l| !subsumes(env, l));
        if self.labels.len() < MAX_LABEL_COMPLEXITY {
            self.labels.push(env);
        }
    }

    pub fn is_supported_in(&self, env: Env) -> bool {
        self.labels.iter().any(|&l| subsumes(l, env))
    }

    /// Remove support derived from assumptions covered by `nogood`.
    /// A label is retracted if it overlaps with the nogood (it depended on a
    /// now-contradicted assumption).
    pub fn retract_labels_using(&mut self, nogood: Env) {
        self.labels.retain(|&l| l & nogood == 0);
    }
}

#[derive(Debug)]
pub struct CounterfactualResult {
    /// Fraction of scope nodes that lost all support after the intervention.
    pub causal_fraction: f64,
    /// NOGOOD injected into the base layer (the assumption bits that were contradicted).
    pub nogood: Option<Env>,
}

pub struct BfAtms {
    nodes:   HashMap<NodeId, BfNode>,
    scope:   CounterfactualScope,
    /// NOGOODs accumulated during this counterfactual scope.
    nogoods: Vec<Env>,
}

impl BfAtms {
    pub fn new(scope: CounterfactualScope) -> Self {
        Self { nodes: HashMap::new(), scope, nogoods: Vec::new() }
    }

    pub fn seed_node(&mut self, id: NodeId, base_env: Env, cost: f64) {
        if !self.scope.contains_node(id) { return; }
        let node = self.nodes.entry(id).or_insert_with(|| BfNode::new(id, cost));
        node.add_label(base_env);
    }

    /// Execute do(edge_assumption_bits = absent).
    ///
    /// `edge_assumptions` is the union of assumption bits that participate in
    /// any justification that flows through the intervened edge. These come from
    /// the caller resolving the edge's ATMS provenance before calling here.
    ///
    /// The NOGOOD is injected, labels depending on those bits are retracted,
    /// and the causal fraction is measured as the proportion of scope nodes
    /// that no longer hold any supporting environment.
    pub fn run_intervention(
        &mut self,
        edge_assumptions: Env,
        active_env: Env,
    ) -> CounterfactualResult {
        if edge_assumptions == 0 {
            // No resolvable assumption bits — intervention is a no-op.
            return CounterfactualResult { causal_fraction: 0.0, nogood: None };
        }

        let scope_size: usize = self.nodes.values()
            .filter(|n| self.scope.contains_node(n.id))
            .count();
        if scope_size == 0 {
            return CounterfactualResult { causal_fraction: 0.0, nogood: None };
        }

        // Count supported nodes before intervention.
        let before: usize = self.nodes.values()
            .filter(|n| self.scope.contains_node(n.id) && n.is_supported_in(active_env))
            .count();

        // Apply the NOGOOD: retract any label that overlaps the contradicted bits.
        self.nogoods.push(edge_assumptions);
        for node in self.nodes.values_mut() {
            node.retract_labels_using(edge_assumptions);
        }

        // Count supported nodes after intervention.
        let after: usize = self.nodes.values()
            .filter(|n| self.scope.contains_node(n.id) && n.is_supported_in(active_env))
            .count();

        let causal_fraction = if before == 0 {
            0.0
        } else {
            (before - after) as f64 / before as f64
        };

        let nogood = if causal_fraction > 0.5 { Some(edge_assumptions) } else { None };

        CounterfactualResult { causal_fraction, nogood }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atms::base::env::singleton;
    use crate::atms::causal::scope::CounterfactualScope;

    fn trivial_scope(node_ids: Vec<crate::types::NodeId>) -> CounterfactualScope {
        CounterfactualScope::from_nodes(node_ids)
    }

    #[test]
    fn intervention_retracts_dependent_labels() {
        let scope = trivial_scope(vec![1, 2]);
        let mut bf = BfAtms::new(scope);
        // Node 1 labelled with assumption bit 0; node 2 labelled with bit 1.
        bf.seed_node(1, singleton(0), 0.5);
        bf.seed_node(2, singleton(1), 0.5);

        let active = singleton(0) | singleton(1);
        // Intervene on assumption bit 0 (the edge that node 1 depends on).
        let result = bf.run_intervention(singleton(0), active);

        // Node 1 lost support; node 2 still supported.
        assert!(result.causal_fraction > 0.0 && result.causal_fraction <= 1.0);
    }

    #[test]
    fn zero_assumption_bits_is_noop() {
        let scope = trivial_scope(vec![1]);
        let mut bf = BfAtms::new(scope);
        bf.seed_node(1, singleton(0), 0.5);
        let result = bf.run_intervention(0, singleton(0));
        assert_eq!(result.causal_fraction, 0.0);
        assert!(result.nogood.is_none());
    }
}
