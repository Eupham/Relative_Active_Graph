//! Variance-Adaptive Threshold (VAT).
//! θ_α(d, t+1) = θ_min + (θ_max − θ_min) × (1 − σ²_d(t) / σ²_max(d))
//!
//! High variance → explore (low θ_α → more nodes activated).
//! Low variance  → exploit (high θ_α → fewer nodes, tighter focus).
//!
//! This is NOT VDBE (Thrun 1992, "The role of exploration in learning control").
//! The formula most closely resembles:
//!   Kaelbling (1993), "Learning to Achieve Goals", IJCAI-93 (Interval Estimation)
//! or the UCB1 exploration bonus:
//!   Auer, Cesa-Bianchi & Fischer (2002), "Finite-time Analysis of the
//!   Multiarmed Bandit Problem", Machine Learning 47:235–256.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::types::TRDId;
use super::performance_ema::PerfRegistry;

pub const THETA_ALPHA_MIN: f64 = 0.1;
pub const THETA_ALPHA_MAX: f64 = 0.8;
/// θ_ρ = θ_α × 0.95 (edge threshold slightly below node threshold).
pub const RHO_ALPHA_RATIO: f64 = 0.95;
/// Cold-start value for new TRDs.
pub const THETA_COLD_START: f64 = 0.4;

/// Per-TRD threshold state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrdThresholds {
    pub trd_id:      TRDId,
    pub theta_alpha: f64,
    pub theta_rho:   f64,
}

impl TrdThresholds {
    pub fn cold_start(trd_id: TRDId) -> Self {
        Self {
            trd_id,
            theta_alpha: THETA_COLD_START,
            theta_rho:   THETA_COLD_START * RHO_ALPHA_RATIO,
        }
    }

    /// Recompute from current variance ratio.
    pub fn update(&mut self, variance_ratio: f64) {
        self.theta_alpha = THETA_ALPHA_MIN
            + (THETA_ALPHA_MAX - THETA_ALPHA_MIN) * (1.0 - variance_ratio.clamp(0.0, 1.0));
        self.theta_rho = self.theta_alpha * RHO_ALPHA_RATIO;
    }
}

/// Registry of per-TRD thresholds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdRegistry {
    thresholds: HashMap<TRDId, TrdThresholds>,
}

impl ThresholdRegistry {
    pub fn new() -> Self {
        Self { thresholds: HashMap::new() }
    }

    /// Get thresholds for a TRD, initializing at cold-start if not seen before.
    pub fn get(&mut self, trd_id: TRDId) -> &TrdThresholds {
        self.thresholds.entry(trd_id).or_insert_with(|| TrdThresholds::cold_start(trd_id))
    }

    /// Recompute thresholds from the performance registry.
    pub fn sync(&mut self, perf: &PerfRegistry) {
        for (trd_id, thresh) in self.thresholds.iter_mut() {
            let ratio = perf.variance_ratio(*trd_id);
            thresh.update(ratio);
        }
    }

    /// Convenience: get theta_alpha for a TRD (creating cold start if needed).
    pub fn theta_alpha(&mut self, trd_id: TRDId) -> f64 {
        self.get(trd_id).theta_alpha
    }

    pub fn theta_rho(&mut self, trd_id: TRDId) -> f64 {
        self.get(trd_id).theta_rho
    }

    /// Ensure thresholds diverge between two TRDs under different performance regimes.
    /// Used in experiment 2 validation.
    pub fn are_diverged(&self, trd_a: TRDId, trd_b: TRDId) -> bool {
        let a = self.thresholds.get(&trd_a).map(|t| t.theta_alpha).unwrap_or(THETA_COLD_START);
        let b = self.thresholds.get(&trd_b).map(|t| t.theta_alpha).unwrap_or(THETA_COLD_START);
        (a - b).abs() > 0.1
    }
}

impl Default for ThresholdRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Quality;

    #[test]
    fn stable_trd_exploits() {
        let mut thresh = TrdThresholds::cold_start(0);
        thresh.update(0.02); // near-zero variance ratio → exploit
        assert!(thresh.theta_alpha > 0.6, "expected exploit-mode threshold, got {}", thresh.theta_alpha);
    }

    #[test]
    fn noisy_trd_explores() {
        let mut thresh = TrdThresholds::cold_start(0);
        thresh.update(0.95); // near-max variance → explore
        assert!(thresh.theta_alpha < 0.2, "expected explore-mode threshold, got {}", thresh.theta_alpha);
    }

    #[test]
    fn two_trds_diverge() {
        let mut perf = PerfRegistry::new(0.25);
        let mut reg  = ThresholdRegistry::new();

        // TRD 0: all good → stable
        for _ in 0..40 { perf.update(0, Quality::GOOD); }
        // TRD 1: alternating → noisy
        for i in 0..40u32 { perf.update(1, if i % 2 == 0 { Quality::GOOD } else { Quality::BAD }); }

        reg.get(0);
        reg.get(1);
        reg.sync(&perf);

        let a = reg.theta_alpha(0);
        let b = reg.theta_alpha(1);
        // They should differ because one TRD is stable, the other is noisy.
        assert!((a - b).abs() > 0.0, "thresholds should diverge: a={a} b={b}");
    }
}
