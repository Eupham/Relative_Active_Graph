//! Per-TRD causal transition: N_ready(e, d) formula.
//! An edge class transitions from correlational to causal phase when sufficient
//! dissolved TRs from TRD d have used that edge class.

use std::collections::HashMap;
use crate::types::{EdgeId, TRDId};

/// χ²_{0.025} = 1.96² ≈ 3.84 (two-tailed 95% CI).
const Z_SQUARED: f64 = 3.8416;

/// Attribution precision ε: desired accuracy of attribution estimates.
/// Smaller ε → more samples required before causal transition.
pub const EPSILON: f64 = 0.1;

/// Phase of attribution for an (edge_class, TRD) pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttributionPhase {
    Correlational,
    Causal,
}

/// Per-(edge_class, TRD) attribution state.
#[derive(Clone, Debug)]
pub struct EdgeTrdState {
    pub edge_id:      EdgeId,
    pub trd_id:       TRDId,
    pub phase:        AttributionPhase,
    /// Attribution scores from dissolved TRs using this edge in this TRD.
    scores:           Vec<f64>,
    /// N_ready computed from current score variance.
    pub n_ready:      usize,
}

impl EdgeTrdState {
    pub fn new(edge_id: EdgeId, trd_id: TRDId) -> Self {
        Self {
            edge_id,
            trd_id,
            phase:   AttributionPhase::Correlational,
            scores:  Vec::new(),
            n_ready: 10, // conservative initial estimate
        }
    }

    /// Record a new attribution score from a dissolved TR using this edge in this TRD.
    pub fn record(&mut self, score: f64) {
        self.scores.push(score);
        self.recompute_n_ready();
        if self.scores.len() >= self.n_ready {
            self.phase = AttributionPhase::Causal;
        }
    }

    /// N_ready(e, d) = ⌈ z²_{0.025} × σ²(Δ(e), d) / ε² ⌉
    fn recompute_n_ready(&mut self) {
        if self.scores.len() < 2 { return; }
        let n = self.scores.len() as f64;
        let mean = self.scores.iter().sum::<f64>() / n;
        let var  = self.scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
        // Minimum 20 samples before causal transition regardless of computed N_ready.
        self.n_ready = ((Z_SQUARED * var / (EPSILON * EPSILON)).ceil() as usize).max(20);
    }

    pub fn correlational_score(&self) -> f64 {
        if self.scores.is_empty() { return 0.5; }
        self.scores.iter().sum::<f64>() / self.scores.len() as f64
    }

    pub fn sample_count(&self) -> usize { self.scores.len() }
}

/// Registry of all (edge, TRD) attribution states.
pub struct CausalTransitionRegistry {
    states: HashMap<(EdgeId, TRDId), EdgeTrdState>,
}

impl CausalTransitionRegistry {
    pub fn new() -> Self {
        Self { states: HashMap::new() }
    }

    pub fn record(&mut self, edge_id: EdgeId, trd_id: TRDId, score: f64) {
        self.states.entry((edge_id, trd_id))
            .or_insert_with(|| EdgeTrdState::new(edge_id, trd_id))
            .record(score);
    }

    pub fn phase(&self, edge_id: EdgeId, trd_id: TRDId) -> AttributionPhase {
        self.states.get(&(edge_id, trd_id))
            .map_or(AttributionPhase::Correlational, |s| s.phase)
    }

    pub fn correlational_score(&self, edge_id: EdgeId, trd_id: TRDId) -> f64 {
        self.states.get(&(edge_id, trd_id)).map_or(0.5, |s| s.correlational_score())
    }

    pub fn n_ready(&self, edge_id: EdgeId, trd_id: TRDId) -> usize {
        self.states.get(&(edge_id, trd_id)).map_or(10, |s| s.n_ready)
    }

    /// Is this (edge, TRD) pair in causal phase?
    pub fn is_causal(&self, edge_id: EdgeId, trd_id: TRDId) -> bool {
        self.phase(edge_id, trd_id) == AttributionPhase::Causal
    }
}

impl Default for CausalTransitionRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitions_to_causal_with_enough_samples() {
        let mut s = EdgeTrdState::new(1, 0);
        // Feed many uniform samples (low variance → small N_ready)
        for _ in 0..50 {
            s.record(0.8);
        }
        assert_eq!(s.phase, AttributionPhase::Causal);
    }

    #[test]
    fn stays_correlational_with_few_samples() {
        let mut s = EdgeTrdState::new(1, 0);
        s.record(0.9);
        s.record(0.1); // high variance → large N_ready (> 20 minimum)
        // With only 2 samples, must be < N_ready (minimum 20).
        assert_eq!(s.phase, AttributionPhase::Correlational);
        assert!(s.n_ready >= 20);
    }

    #[test]
    fn phase_independent_per_trd() {
        let mut reg = CausalTransitionRegistry::new();
        // TRD 0: many samples → causal
        for _ in 0..50 { reg.record(1, 0, 0.8); }
        // TRD 1: few samples → still correlational
        reg.record(1, 1, 0.7);
        assert!(reg.is_causal(1, 0));
        assert!(!reg.is_causal(1, 1));
    }
}
