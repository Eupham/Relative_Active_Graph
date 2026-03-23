//! Minimal stub — algebraic normalizer and weight helpers.
//! Referenced by lifting.rs and graphica_adapter.rs.

/// Placeholder algebraic normalizer (no-op implementation).
pub struct AlgebraicNormalizer;

impl AlgebraicNormalizer {
    pub fn new() -> Self { Self }
}

impl Default for AlgebraicNormalizer { fn default() -> Self { Self::new() } }

/// Apply a named algebraic transform to a weight. Returns weight unchanged (stub).
pub fn apply_named_transform(_normalizer: &AlgebraicNormalizer, _name: &str, weight: f32) -> f32 {
    weight
}

/// Normalize a path weight sequence as geometric mean.
pub fn normalize_path_weight(weights: &[f32]) -> f32 {
    if weights.is_empty() { return 0.5; }
    let log_sum: f64 = weights.iter().map(|&w| (w.max(1e-9) as f64).ln()).sum();
    (log_sum / weights.len() as f64).exp() as f32
}
