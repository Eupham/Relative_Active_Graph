//! Counterfactual reasoning: "what would have happened if edge e had been absent?"
//! Combines SCM intervention + BF-ATMS scope + causal bootstrapping into a unified API.

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, TRDId, Env, Quality};
use crate::causal::{
    scm::Scm,
    bootstrap::{CausalBootstrapper, QualitySample, BootstrapResult, compute_causal_delta},
    intervention::{do_absent, InterventionResult},
};
use crate::adaptive::CausalTransitionRegistry;
use crate::arg::ArgGraph;

/// Δ(e, d) = combined attribution score for edge e in TRD d.
#[derive(Debug)]
pub struct AttributionDelta {
    pub edge_id:               EdgeId,
    pub trd_id:                TRDId,
    pub delta:                 f64,
    /// True when this TRD has accumulated sufficient samples (§13).
    /// This measures sample count readiness, NOT causal identification.
    pub has_sufficient_samples: bool,
}

/// The counterfactual reasoner: orchestrates SCM + BF-ATMS + bootstrap.
pub struct CounterfactualReasoner {
    pub scm:          Scm,
    pub bootstrapper: CausalBootstrapper,
    pub transition:   CausalTransitionRegistry,
    active_env:       Env,
    budget:           usize,
}

impl CounterfactualReasoner {
    pub fn new(active_env: Env, seed: u64) -> Self {
        Self {
            scm:          Scm::new(),
            bootstrapper: CausalBootstrapper::new(seed),
            transition:   CausalTransitionRegistry::new(),
            active_env,
            budget:       50,
        }
    }

    /// Record a dissolved TR's contribution to an edge in a TRD.
    pub fn record_dissolved_tr(
        &mut self,
        tr_id:      u64,
        trd_id:     TRDId,
        quality:    Quality,
        edge_ids:   Vec<EdgeId>,
    ) {
        let q = quality.as_f64();
        let sample = QualitySample { tr_id, trd_id, quality: q, edge_contributions: edge_ids.clone() };
        self.bootstrapper.add_sample(sample);
        for eid in edge_ids {
            self.transition.record(eid, trd_id, q);
        }
    }

    /// Compute Δ(e, d) for edge `edge_id` in TRD `trd_id`.
    /// Uses causal phase if N_ready reached, otherwise correlational.
    pub fn compute_delta(
        &mut self,
        edge_id:  EdgeId,
        trd_id:   TRDId,
        graph:    &ArgGraph,
    ) -> AttributionDelta {
        let has_sufficient = self.transition.is_causal(edge_id, trd_id);

        let delta = if has_sufficient {
            // Sufficient samples: bootstrapped attributional score with CI (§13)
            let result    = self.bootstrapper.estimate_attributional_score(edge_id, trd_id);
            let frequency = self.bootstrapper.frequency_in_trd(edge_id, trd_id);
            compute_causal_delta(&result, frequency)
        } else {
            // Insufficient samples: correlational score only
            self.transition.correlational_score(edge_id, trd_id)
        };

        AttributionDelta { edge_id, trd_id, delta, has_sufficient_samples: has_sufficient }
    }

    /// Type consistency: fraction of contexts where this edge's modal mode matched the derivation.
    fn estimate_type_consistency(&self, edge_id: EdgeId, graph: &ArgGraph) -> f64 {
        let Some(ei) = graph.edge_indices().find(|&ei| graph[ei].id == edge_id) else {
            return 0.5;
        };
        let edge = &graph[ei];
        let (src_idx, dst_idx) = graph.edge_endpoints(ei).unwrap();
        let src_mode = graph[src_idx].mtlg_type.mode;
        let edge_mode = edge.modal_mode;
        if src_mode == edge_mode { 1.0 } else { 0.0 }
    }

    /// Run do(e=absent) intervention and record the result for future attribution.
    pub fn run_intervention(
        &mut self,
        edge_id: EdgeId,
        graph:   &ArgGraph,
    ) -> InterventionResult {
        do_absent(graph, &self.scm, edge_id, self.active_env, self.budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Quality;

    #[test]
    fn correlational_delta_from_samples() {
        let mut g: ArgGraph = petgraph::stable_graph::StableGraph::new();
        let mut reasoner = CounterfactualReasoner::new(0b11, 42);
        // Feed quality samples for edge 10 in TRD 0
        for _ in 0..5 { reasoner.record_dissolved_tr(1, 0, Quality::GOOD, vec![10]); }
        for _ in 0..5 { reasoner.record_dissolved_tr(2, 0, Quality::BAD, vec![10]); }
        let delta = reasoner.compute_delta(10, 0, &g);
        assert!(!delta.has_sufficient_samples); // not enough samples yet
        assert!((0.0..=1.0).contains(&delta.delta));
    }
}
