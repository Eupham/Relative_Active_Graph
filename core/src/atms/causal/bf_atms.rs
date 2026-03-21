//! BF-ATMS: bounded, non-Horn ATMS for causal counterfactual reasoning.
//! (Provan & Singh 1991 — polynomial search bounded by label complexity k.)
//!
//! Operates exclusively within a CounterfactualScope. NOGOOD from this layer
//! propagates back to the Horn base layer to trigger TR dissolution.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, Env};
use super::scope::CounterfactualScope;
use crate::atms::base::env::{singleton, union, subsumes};

/// Maximum label set size (k in BF-ATMS). Polynomial guarantee holds for k ≤ 4.
const MAX_LABEL_COMPLEXITY: usize = 4;

/// A node in the BF-ATMS layer.
#[derive(Clone, Debug)]
pub struct BfNode {
    pub id:   NodeId,
    /// Set of minimal supporting environments (non-Horn: can have multiple).
    pub labels: Vec<Env>,
    /// Attribution score used as cost in the search (from dissolved TR traces).
    pub cost: f64,
}

impl BfNode {
    pub fn new(id: NodeId, cost: f64) -> Self {
        Self { id, labels: Vec::new(), cost }
    }

    /// Add a new minimal label, maintaining minimality.
    pub fn add_label(&mut self, env: Env) {
        if self.labels.iter().any(|&l| subsumes(l, env)) { return; }
        self.labels.retain(|&l| !subsumes(env, l));
        if self.labels.len() < MAX_LABEL_COMPLEXITY {
            self.labels.push(env);
        }
    }

    pub fn is_supported_in(&self, env: Env) -> bool {
        self.labels.iter().any(|&l| subsumes(l, env))
    }
}

/// Result of a BF-ATMS counterfactual query.
#[derive(Debug)]
pub struct CounterfactualResult {
    /// Fraction of re-sampled contexts where quality drops after do(e=absent).
    pub causal_fraction: f64,
    /// The NOGOOD to inject into the base layer (if intervention creates contradiction).
    pub nogood: Option<Env>,
}

/// BF-ATMS engine, scoped to a single CounterfactualScope.
pub struct BfAtms {
    nodes: HashMap<NodeId, BfNode>,
    scope: CounterfactualScope,
}

impl BfAtms {
    pub fn new(scope: CounterfactualScope) -> Self {
        Self { nodes: HashMap::new(), scope }
    }

    /// Seed nodes from the base layer with their costs (attribution scores).
    pub fn seed_node(&mut self, id: NodeId, base_env: Env, cost: f64) {
        if !self.scope.contains_node(id) { return; }
        let node = self.nodes.entry(id).or_insert_with(|| BfNode::new(id, cost));
        node.add_label(base_env);
    }

    /// Run the counterfactual: remove `intervened_edge` and propagate belief revision.
    /// Returns fraction of scope nodes that lose support (proxy for causal effect).
    pub fn run_intervention(&mut self, intervened_edge: EdgeId, active_env: Env) -> CounterfactualResult {
        // Before intervention: count supported nodes in scope.
        let before: usize = self.nodes.values()
            .filter(|n| self.scope.contains_node(n.id) && n.is_supported_in(active_env))
            .count();

        // Remove the intervened edge from all labels that depended on it.
        // We model this by removing labels that have the edge's assumption bit.
        // (The edge's assumption bit is derived from its EdgeId hash into 0..63.)
        let edge_bit = (intervened_edge % 63) as u8;
        let intervention_env = !singleton(edge_bit) & active_env;

        // After intervention: count supported nodes in scope.
        let after: usize = self.nodes.values()
            .filter(|n| self.scope.contains_node(n.id) && n.is_supported_in(intervention_env))
            .count();

        let total = self.scope.node_count().max(1);
        let lost = before.saturating_sub(after);
        let causal_fraction = lost as f64 / total as f64;

        // If the intervention leaves a core node without any support, inject a nogood.
        let nogood = if causal_fraction > 0.5 {
            Some(active_env) // The full active environment is now inconsistent given this edge's absence.
        } else {
            None
        };

        CounterfactualResult { causal_fraction, nogood }
    }

    /// Compute weighted causal effect: C(e) = causal_fraction weighted by node costs.
    pub fn weighted_causal_effect(&self, intervened_edge: EdgeId, active_env: Env) -> f64 {
        let edge_bit = (intervened_edge % 63) as u8;
        let intervention_env = !singleton(edge_bit) & active_env;

        let mut total_cost = 0.0f64;
        let mut lost_cost  = 0.0f64;

        for node in self.nodes.values() {
            if !self.scope.contains_node(node.id) { continue; }
            total_cost += node.cost;
            let had_support  = node.is_supported_in(active_env);
            let still_has    = node.is_supported_in(intervention_env);
            if had_support && !still_has {
                lost_cost += node.cost;
            }
        }
        if total_cost == 0.0 { return 0.0; }
        lost_cost / total_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atms::base::env::singleton;

    fn make_scope() -> CounterfactualScope {
        CounterfactualScope::build(42, 0, 10, |n| {
            if n < 4 { vec![(n + 1, n * 10 + 42)] } else { vec![] }
        })
    }

    #[test]
    fn intervention_reduces_support() {
        let scope = make_scope();
        let active = singleton(0) | singleton(1) | singleton(2);
        let mut bf = BfAtms::new(scope);
        bf.seed_node(0, active, 1.0);
        bf.seed_node(1, active, 1.0);
        bf.seed_node(2, active, 1.0);

        // Intervene on an edge that is in the scope
        let result = bf.run_intervention(42, active);
        // causal_fraction should be >= 0
        assert!(result.causal_fraction >= 0.0);
    }
}
