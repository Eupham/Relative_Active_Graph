//! TRD-relative causal bootstrapping (Little & Badawy 2020).
//! Resamples context with do(e=absent) and estimates the ATE with a 95% CI.
//! An edge's causal effect is considered reliable only when CI lower bound > 0.

use rand::prelude::*;
use crate::types::{NodeId, EdgeId, TRDId, Quality};

pub const N_BOOTSTRAP: usize = 200;
const CI_LOWER_PERCENTILE: f64 = 2.5;
const CI_UPPER_PERCENTILE: f64 = 97.5;

#[derive(Clone, Debug)]
pub struct QualitySample {
    pub tr_id:               u64,
    pub trd_id:              TRDId,
    pub quality:             f64,
    pub edge_contributions:  Vec<EdgeId>,
}

/// Bootstrap estimate of the causal effect of edge `e` in TRD `d`.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Mean quality drop across all bootstrap resamples (point estimate of ATE).
    pub point_estimate: f64,
    /// 2.5th percentile of the bootstrap distribution of quality drops.
    pub ci_lower:       f64,
    /// 97.5th percentile of the bootstrap distribution of quality drops.
    pub ci_upper:       f64,
    pub n_resamples:    usize,
    /// True only when the CI lower bound exceeds zero — i.e., the effect is
    /// reliably positive at the 95% level under the bootstrap distribution.
    pub is_reliable:    bool,
}

impl BootstrapResult {
    /// Causal effect is actionable only when reliably positive.
    pub fn causal_fraction(&self) -> f64 {
        if self.is_reliable { self.point_estimate } else { 0.0 }
    }
}

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

    /// Estimate the causal effect of `edge_id` in `trd_id` as a bootstrapped ATE.
    ///
    /// Each resample computes the mean quality drop observed when samples using
    /// `edge_id` are excluded (simulating do(e=absent)). The distribution of
    /// these per-resample drops is used to compute a 95% CI.
    ///
    /// Returns a zero-effect result when insufficient data is available.
    pub fn estimate_causal_effect(
        &mut self,
        edge_id: EdgeId,
        trd_id:  TRDId,
    ) -> BootstrapResult {
        let trd_samples: Vec<&QualitySample> = self.samples.iter()
            .filter(|s| s.trd_id == trd_id)
            .collect();

        if trd_samples.len() < 10 {
            return BootstrapResult {
                point_estimate: 0.0,
                ci_lower:       0.0,
                ci_upper:       0.0,
                n_resamples:    0,
                is_reliable:    false,
            };
        }

        let baseline: f64 = trd_samples.iter().map(|s| s.quality).sum::<f64>()
            / trd_samples.len() as f64;

        let mut per_resample_drops = Vec::with_capacity(N_BOOTSTRAP);

        for _ in 0..N_BOOTSTRAP {
            let n = trd_samples.len();
            let resampled: Vec<&QualitySample> = (0..n)
                .map(|_| trd_samples[self.rng.gen_range(0..n)])
                .collect();

            let without_edge: Vec<_> = resampled.iter()
                .filter(|s| !s.edge_contributions.contains(&edge_id))
                .collect();

            if without_edge.is_empty() {
                per_resample_drops.push(0.0);
                continue;
            }

            let q_without: f64 = without_edge.iter().map(|s| s.quality).sum::<f64>()
                / without_edge.len() as f64;

            per_resample_drops.push(baseline - q_without);
        }

        per_resample_drops.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let point_estimate = per_resample_drops.iter().sum::<f64>() / N_BOOTSTRAP as f64;
        let ci_lower = percentile(&per_resample_drops, CI_LOWER_PERCENTILE);
        let ci_upper = percentile(&per_resample_drops, CI_UPPER_PERCENTILE);
        let is_reliable = ci_lower > 0.0;

        BootstrapResult { point_estimate, ci_lower, ci_upper, n_resamples: N_BOOTSTRAP, is_reliable }
    }

    /// Frequency score: fraction of TRD samples in which edge_id participated.
    pub fn frequency_in_trd(&self, edge_id: EdgeId, trd_id: TRDId) -> f64 {
        let trd: Vec<_> = self.samples.iter().filter(|s| s.trd_id == trd_id).collect();
        if trd.is_empty() { return 0.0; }
        let with_edge = trd.iter().filter(|s| s.edge_contributions.contains(&edge_id)).count();
        with_edge as f64 / trd.len() as f64
    }
}

/// Extract the p-th percentile from a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Combined Δ(e, d): point estimate when reliable, frequency-weighted otherwise.
pub fn compute_causal_delta(result: &BootstrapResult, frequency: f64) -> f64 {
    if result.is_reliable {
        result.point_estimate * frequency
    } else {
        // Correlational proxy: frequency alone, discounted by uncertainty.
        let uncertainty = (result.ci_upper - result.ci_lower).max(0.0);
        frequency * (1.0 - uncertainty.min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(tr_id: u64, trd_id: TRDId, quality: f64, edges: Vec<EdgeId>) -> QualitySample {
        QualitySample { tr_id, trd_id, quality, edge_contributions: edges }
    }

    #[test]
    fn reliable_effect_detected() {
        let mut b = CausalBootstrapper::new(42);
        // 20 samples: edge 1 always present, quality always 1.0.
        // 20 samples: edge 1 absent, quality always 0.0.
        for i in 0..20 {
            b.add_sample(make_sample(i, 0, 1.0, vec![1]));
        }
        for i in 20..40 {
            b.add_sample(make_sample(i, 0, 0.0, vec![]));
        }
        let result = b.estimate_causal_effect(1, 0);
        // Removing edge 1 drops quality from 0.5 baseline (mixed) to ~0.0 in without-edge set.
        // The CI lower bound should be positive.
        assert!(result.point_estimate > 0.0, "expected positive ATE: {:?}", result);
    }

    #[test]
    fn insufficient_data_returns_zero_effect() {
        let mut b = CausalBootstrapper::new(42);
        b.add_sample(make_sample(0, 0, 0.9, vec![1]));
        let result = b.estimate_causal_effect(1, 0);
        assert!(!result.is_reliable);
        assert_eq!(result.causal_fraction(), 0.0);
    }

    #[test]
    fn percentile_boundary_cases() {
        let sorted = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        assert_eq!(percentile(&sorted, 0.0),   0.1);
        assert_eq!(percentile(&sorted, 100.0), 0.5);
        assert_eq!(percentile(&[], 50.0),      0.0);
    }

    #[test]
    fn frequency_in_trd_correct() {
        let mut b = CausalBootstrapper::new(42);
        b.add_sample(make_sample(1, 0, 1.0, vec![10]));
        b.add_sample(make_sample(2, 0, 0.0, vec![10]));
        b.add_sample(make_sample(3, 0, 1.0, vec![]));
        let freq = b.frequency_in_trd(10, 0);
        assert!((freq - 2.0/3.0).abs() < 0.01, "freq={freq}");
    }
}
