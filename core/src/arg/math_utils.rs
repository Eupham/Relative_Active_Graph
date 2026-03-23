//! Numerica: discrete causal mathematics for the ARG engine.
//! Owns all numerical primitives: normalization, softmax, depth scoring, noise.
//! Uses standard Rust iterators and rand_distr — no ML, no CAS.

use rand_distr::{Poisson, Distribution};

/// Normalize a path-weight sequence via Log-Sum-Exp (smooth maximum in log-space).
///
/// Uses the numerically stable form: LSE(w) = max(w) + ln(Σ exp(wᵢ - max(w)))
///
/// # Edge cases
/// - Empty slice → `0.0` (no evidence, neutral).
pub fn normalize_path_weight(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }
    let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = weights.iter().map(|&w| (w - max_w).exp()).sum();
    max_w + sum_exp.ln()
}

/// Numerically stable softmax over attribution scores.
///
/// Returns a probability vector in the same order as `logits`.
/// Uses max-subtraction trick for numerical stability.
/// Empty input → empty output.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max_score = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>() + 1e-12;
    exps.into_iter().map(|e| e / sum).collect()
}

/// Compute ARG node depth from path density, attribution, type complexity,
/// and causal structure.
///
/// depth = 0.4·path_density + 0.3·attribution + 0.2·ln(arity+1) + 0.1·ln(parents+1)
pub fn compute_depth(path_density: f32, attribution_score: f32, type_arity: u8, causal_parent_count: usize) -> f32 {
    let type_complexity = (type_arity as f32 + 1.0).ln();
    let causal_factor   = (causal_parent_count as f32 + 1.0).ln();
    (path_density * 0.4) + (attribution_score * 0.3) + (type_complexity * 0.2) + (causal_factor * 0.1)
}

/// Sample the number of discrete perturbation events from a Poisson(λ) distribution.
///
/// # Parameters
/// - `lambda`: expected event count per cycle (§17 perturbation rate).
/// - `rng`: any `rand::Rng` implementation (thread-local, seeded, etc.).
///
/// # Edge cases
/// - `lambda == 0.0` → returns `0` immediately without invoking the sampler
///   (fully deterministic baseline, no floating-point allocation).
pub fn sample_discrete_noise(lambda: f64, rng: &mut impl rand::Rng) -> u64 {
    if lambda == 0.0 {
        return 0;
    }
    Poisson::new(lambda)
        .expect("valid Poisson parameter: lambda must be > 0")
        .sample(rng) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lse_empty_returns_zero() {
        assert_eq!(normalize_path_weight(&[]), 0.0);
    }

    #[test]
    fn lse_single_element_is_identity() {
        let w = 2.5_f32;
        let result = normalize_path_weight(&[w]);
        // LSE([w]) = w + ln(exp(0)) = w + ln(1) = w
        assert!((result - w).abs() < 1e-5, "got {result}");
    }

    #[test]
    fn lse_numerically_stable_large_values() {
        // Would overflow with naive exp(88) in f32; LSE subtracts max first.
        let weights = [80.0_f32, 88.0, 84.0];
        let result = normalize_path_weight(&weights);
        assert!(result.is_finite(), "LSE must be finite for large weights");
        // LSE({80, 88, 84}) should be close to 88 (dominated by max element).
        assert!(result > 87.0 && result < 90.0,
            "LSE should be near max element (~88), got {result}");
    }

    #[test]
    fn poisson_zero_lambda_returns_zero() {
        let mut rng = rand::thread_rng();
        assert_eq!(sample_discrete_noise(0.0, &mut rng), 0);
    }

    #[test]
    fn poisson_nonzero_lambda_produces_samples() {
        let mut rng = rand::thread_rng();
        // Over 1000 draws from Poisson(5.0), at least one must be nonzero
        // with astronomically high probability.
        let any_nonzero = (0..1000).any(|_| sample_discrete_noise(5.0, &mut rng) > 0);
        assert!(any_nonzero);
    }
}
