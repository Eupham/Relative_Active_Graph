//! TRD-relative causal bootstrapping (Little & Badawy 2020).
//! Resamples context with do(e=absent) and measures quality drop.
//! C(e) = fraction of resamples where Q drops after the intervention.

use rand::prelude::*;
use crate::types::{NodeId, EdgeId, TRDId, Quality};
use crate::causal::scm::Scm;

pub const N_BOOTSTRAP: usize = 100;

/// A quality measurement from a dissolved TR in this TRD.
#[derive(Clone, Debug)]
pub struct QualitySample {
    pub tr_id:   u64,
    pub trd_id:  TRDId,
    pub quality: f64,
    /// Which edge IDs contributed to this TR.
    pub edge_contributions: Vec<EdgeId>,
}

/// Bootstrap estimator for causal effect of edge `e` in TRD `d`.
pub struct CausalBootstrapper {
    samples: Vec<QualitySample>,
    rng:     StdRng,
}

impl CausalBootstrapper {
    pub fn new(seed: u64) -> Self {
        Self { samples: Vec::new(), rng: StdRng::seed_from_u64(seed) }
    }

    pub fn add_sample(&mut self, sample: QualitySample) {
        self.samples.push(sample);
    }

    /// Estimate C(e, d): fraction of bootstrap resamples in which removing edge `e`
    /// from TRD `d` causes quality to drop.
    pub fn estimate_causal_fraction(&mut self, edge_id: EdgeId, trd_id: TRDId) -> f64 {
        let trd_samples: Vec<&QualitySample> = self.samples.iter()
            .filter(|s| s.trd_id == trd_id)
            .collect();
        if trd_samples.len() < 5 { return 0.0; } // insufficient data

        let baseline_quality: f64 = trd_samples.iter().map(|s| s.quality).sum::<f64>()
            / trd_samples.len() as f64;

        let mut drop_count = 0usize;
        for _ in 0..N_BOOTSTRAP {
            // Resample with replacement.
            let resampled: Vec<_> = (0..trd_samples.len())
                .map(|_| trd_samples[self.rng.gen_range(0..trd_samples.len())])
                .collect();
            // Remove samples that used edge_id (simulating do(e=absent)).
            let without_edge: Vec<_> = resampled.iter()
                .filter(|s| !s.edge_contributions.contains(&edge_id))
                .collect();
            if without_edge.is_empty() { continue; }
            let quality_without: f64 = without_edge.iter().map(|s| s.quality).sum::<f64>()
                / without_edge.len() as f64;
            if quality_without < baseline_quality - 0.05 {
                drop_count += 1;
            }
        }
        drop_count as f64 / N_BOOTSTRAP as f64
    }

    /// Frequency score: freq_successful_in_d(e) / freq_total_in_d(e).
    pub fn frequency_score(&self, edge_id: EdgeId, trd_id: TRDId) -> f64 {
        let using_edge: Vec<_> = self.samples.iter()
            .filter(|s| s.trd_id == trd_id && s.edge_contributions.contains(&edge_id))
            .collect();
        if using_edge.is_empty() { return 0.0; }
        let successful = using_edge.iter().filter(|s| s.quality >= 0.5).count();
        successful as f64 / using_edge.len() as f64
    }
}

/// Combined Δ(e) in causal phase:
/// Δ(e) = α·C(e) + β·frequency + γ·type_consistency
pub fn compute_causal_delta(
    causal_fraction:  f64,
    frequency_score:  f64,
    type_consistency: f64,
    alpha: f64,  // weight for causal fraction (default: 0.5)
    beta:  f64,  // weight for frequency (default: 0.3)
    gamma: f64,  // weight for type consistency (default: 0.2)
) -> f64 {
    (alpha * causal_fraction + beta * frequency_score + gamma * type_consistency).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(tr_id: u64, trd_id: TRDId, quality: f64, edges: Vec<EdgeId>) -> QualitySample {
        QualitySample { tr_id, trd_id, quality, edge_contributions: edges }
    }

    #[test]
    fn frequency_score_correct() {
        let mut b = CausalBootstrapper::new(42);
        b.add_sample(make_sample(1, 0, 1.0, vec![10]));
        b.add_sample(make_sample(2, 0, 0.0, vec![10]));
        b.add_sample(make_sample(3, 0, 1.0, vec![10]));
        let score = b.frequency_score(10, 0);
        assert!((score - 2.0/3.0).abs() < 0.01, "score={score}");
    }

    #[test]
    fn causal_fraction_with_no_drop() {
        let mut b = CausalBootstrapper::new(42);
        // Add samples that don't use edge 99 — removing it has no effect.
        for i in 0..20 {
            b.add_sample(make_sample(i, 0, 1.0, vec![1, 2, 3]));
        }
        let frac = b.estimate_causal_fraction(99, 0); // edge 99 not used
        // Without edge 99, the same samples exist → no drop
        assert!(frac == 0.0, "fraction should be 0 when edge not used: {frac}");
    }

    #[test]
    fn combined_delta_in_bounds() {
        let delta = compute_causal_delta(0.8, 0.6, 0.9, 0.5, 0.3, 0.2);
        assert!((0.0..=1.0).contains(&delta));
    }
}
