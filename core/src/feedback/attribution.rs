//! TRD-relative attribution: Δ(e) per edge per TRD.
//! Correlational phase: frequency-based.
//! Causal phase: bootstrapped causal fraction + frequency + type consistency.

use std::collections::HashMap;
use crate::types::{EdgeId, TRDId, Quality};
use crate::adaptive::CausalTransitionRegistry;
use crate::causal::bootstrap::{CausalBootstrapper, QualitySample, compute_causal_delta};
use crate::arg::ArgGraph;

/// Attribution score for one edge in one TRD.
#[derive(Clone, Debug)]
pub struct EdgeAttribution {
    pub edge_id:       EdgeId,
    pub trd_id:        TRDId,
    /// The computed Δ(e) ∈ [0,1].
    pub delta:         f64,
    pub is_causal:     bool,
    pub sample_count:  usize,
}

/// The attribution engine: computes Δ(e, d) for all edges across all TRDs.
pub struct AttributionEngine {
    /// (edge_id, trd_id) → (sum_quality, count_total, count_successful)
    freq_table:   HashMap<(EdgeId, TRDId), (f64, usize, usize)>,
    pub bootstrapper: CausalBootstrapper,
    pub transition:   CausalTransitionRegistry,
}

impl AttributionEngine {
    pub fn new(seed: u64) -> Self {
        Self {
            freq_table:   HashMap::new(),
            bootstrapper: CausalBootstrapper::new(seed),
            transition:   CausalTransitionRegistry::new(),
        }
    }

    /// Record a dissolved TR's edges with its quality score.
    pub fn record(
        &mut self,
        tr_id:    u64,
        trd_id:   TRDId,
        quality:  Quality,
        edge_ids: &[EdgeId],
    ) {
        let q = quality.as_f64();
        let sample = QualitySample {
            tr_id,
            trd_id,
            quality: q,
            edge_contributions: edge_ids.to_vec(),
        };
        self.bootstrapper.add_sample(sample);

        for &eid in edge_ids {
            let entry = self.freq_table.entry((eid, trd_id)).or_insert((0.0, 0, 0));
            entry.0 += q;
            entry.1 += 1;
            if q >= 0.5 { entry.2 += 1; }
            self.transition.record(eid, trd_id, q);
        }
    }

    /// Compute Δ(e, d) for a given edge and TRD.
    pub fn compute(
        &mut self,
        edge_id:  EdgeId,
        trd_id:   TRDId,
        graph:    &ArgGraph,
    ) -> EdgeAttribution {
        let is_causal = self.transition.is_causal(edge_id, trd_id);
        let sample_count = self.freq_table
            .get(&(edge_id, trd_id))
            .map_or(0, |e| e.1);

        let delta = if is_causal {
            let causal_frac = self.bootstrapper.estimate_causal_fraction(edge_id, trd_id);
            let freq_score  = self.correlational_score(edge_id, trd_id);
            let type_cons   = self.type_consistency_score(edge_id, graph);
            compute_causal_delta(causal_frac, freq_score, type_cons, 0.5, 0.3, 0.2)
        } else {
            self.correlational_score(edge_id, trd_id)
        };

        EdgeAttribution { edge_id, trd_id, delta, is_causal, sample_count }
    }

    fn correlational_score(&self, edge_id: EdgeId, trd_id: TRDId) -> f64 {
        match self.freq_table.get(&(edge_id, trd_id)) {
            Some(&(sum, count, _)) if count > 0 => sum / count as f64,
            _ => 0.5,
        }
    }

    fn type_consistency_score(&self, edge_id: EdgeId, graph: &ArgGraph) -> f64 {
        let Some(ei) = graph.edge_indices().find(|&i| graph[i].id == edge_id) else {
            return 0.5;
        };
        let (src, _) = graph.edge_endpoints(ei).unwrap();
        if graph[src].mtlg_type.mode == graph[ei].modal_mode { 1.0 } else { 0.0 }
    }

    /// Compute attributions for all tracked edges in the graph.
    pub fn compute_all(
        &mut self,
        trd_id: TRDId,
        graph:  &ArgGraph,
    ) -> Vec<EdgeAttribution> {
        let edge_ids: Vec<EdgeId> = graph.edge_indices()
            .map(|ei| graph[ei].id)
            .collect();
        edge_ids.iter().map(|&eid| self.compute(eid, trd_id, graph)).collect()
    }
}

impl Default for AttributionEngine {
    fn default() -> Self { Self::new(0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlational_score_correct() {
        let mut engine = AttributionEngine::new(42);
        engine.record(1, 0, Quality::Good, &[10]);
        engine.record(2, 0, Quality::Good, &[10]);
        engine.record(3, 0, Quality::Bad,  &[10]);
        let g: ArgGraph = petgraph::stable_graph::StableGraph::new();
        let attr = engine.compute(10, 0, &g);
        // 2 good (1.0) + 1 bad (0.0) = 2/3 ≈ 0.667
        assert!((attr.delta - 2.0/3.0).abs() < 0.01, "delta={}", attr.delta);
    }
}
