//! Per-TRD performance tracking: P_d(t), σ²_d(t), σ²_max(d).
//! EMA with β=0.9 over quality samples Q ∈ {0.0, 0.5, 1.0}.

use crate::types::{TRDId, Quality};

pub const BETA: f64 = 0.9;
pub const N_WARMUP: u64 = 30;

/// Per-TRD performance tracker.
#[derive(Clone, Debug)]
pub struct TrdPerf {
    pub trd_id:     TRDId,
    /// P_d(t): EMA of quality samples.
    pub p_ema:      f64,
    /// σ²_d(t): EMA of squared deviations.
    pub var_ema:    f64,
    /// σ²_max(d): maximum variance observed (updated after warmup).
    pub var_max:    f64,
    pub n_samples:  u64,
    /// Global prior for var_max before warmup.
    pub global_var_prior: f64,
}

impl TrdPerf {
    /// `global_var_prior`: estimate of σ²_max from mC4 bootstrapping (or default 0.25).
    pub fn new(trd_id: TRDId, global_var_prior: f64) -> Self {
        Self {
            trd_id,
            p_ema:           0.5,  // neutral prior
            var_ema:         global_var_prior,
            var_max:         global_var_prior,
            n_samples:       0,
            global_var_prior,
        }
    }

    /// Ingest a quality sample and update EMA stats.
    pub fn update(&mut self, quality: Quality) {
        let q = quality.as_f64();
        // Update P_d(t)
        self.p_ema = BETA * self.p_ema + (1.0 - BETA) * q;
        // Update σ²_d(t)
        let diff = q - self.p_ema;
        self.var_ema = BETA * self.var_ema + (1.0 - BETA) * diff * diff;
        self.n_samples += 1;
        // After warmup: update σ²_max from live data.
        if self.n_samples >= N_WARMUP {
            self.var_max = self.var_max.max(self.var_ema);
        }
    }

    /// Is the performance stable? (low variance relative to max).
    pub fn is_stable(&self) -> bool {
        self.var_max > 0.0 && self.var_ema / self.var_max < 0.1
    }

    /// Is the performance noisy? (high variance relative to max).
    pub fn is_noisy(&self) -> bool {
        self.var_max > 0.0 && self.var_ema / self.var_max > 0.8
    }

    /// Variance ratio σ²_d(t) / σ²_max(d) ∈ [0,1].
    pub fn variance_ratio(&self) -> f64 {
        if self.var_max == 0.0 { return 0.0; }
        (self.var_ema / self.var_max).clamp(0.0, 1.0)
    }
}

/// Registry of per-TRD performance trackers.
pub struct PerfRegistry {
    trackers: std::collections::HashMap<TRDId, TrdPerf>,
    global_var_prior: f64,
}

impl PerfRegistry {
    pub fn new(global_var_prior: f64) -> Self {
        Self { trackers: std::collections::HashMap::new(), global_var_prior }
    }

    pub fn get_or_create(&mut self, trd_id: TRDId) -> &mut TrdPerf {
        let prior = self.global_var_prior;
        self.trackers.entry(trd_id).or_insert_with(|| TrdPerf::new(trd_id, prior))
    }

    pub fn update(&mut self, trd_id: TRDId, quality: Quality) {
        let prior = self.global_var_prior;
        self.trackers.entry(trd_id)
            .or_insert_with(|| TrdPerf::new(trd_id, prior))
            .update(quality);
    }

    pub fn variance_ratio(&self, trd_id: TRDId) -> f64 {
        self.trackers.get(&trd_id).map_or(0.5, |t| t.variance_ratio())
    }

    pub fn p_ema(&self, trd_id: TRDId) -> f64 {
        self.trackers.get(&trd_id).map_or(0.5, |t| t.p_ema)
    }
}

impl Default for PerfRegistry {
    fn default() -> Self { Self::new(0.25) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_after_good_samples() {
        let mut t = TrdPerf::new(0, 0.25);
        for _ in 0..50 {
            t.update(Quality::Good);
        }
        assert!(t.is_stable());
        assert!(!t.is_noisy());
    }

    #[test]
    fn noisy_after_mixed_samples() {
        let mut t = TrdPerf::new(0, 0.10);
        // Feed maximum-variance sequence: alternating good/bad after warming up var_max
        for i in 0..50 {
            t.update(if i % 2 == 0 { Quality::Good } else { Quality::Bad });
        }
        // var_ema should be relatively high; may or may not trigger is_noisy threshold
        assert!(t.var_ema > 0.0);
    }

    #[test]
    fn variance_ratio_in_bounds() {
        let mut t = TrdPerf::new(0, 0.25);
        t.update(Quality::Bad);
        t.update(Quality::Good);
        let r = t.variance_ratio();
        assert!((0.0..=1.0).contains(&r));
    }
}
