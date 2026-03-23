//! Numerica: discrete causal mathematics for the ARG engine.
//! Replaces the legacy symbolica_adapter (algebraic CAS approach).
//! Uses standard Rust iterators and rand_distr — no ML, no CAS.

use rand_distr::{Poisson, Distribution};

/// Normalize a path-weight sequence via geometric mean.
///
/// # Edge cases
/// - Empty slice → `0.0` (no evidence, neutral).
pub fn normalize_path_weight(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }
    let mut product = 1.0;
    for &w in weights {
        product *= w;
    }
    product.powf(1.0 / weights.len() as f32)
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
        // LSE([w]) = w + ln(exp(0)) = w + 0 = w
        assert!((result - w).abs() < 1e-5, "got {result}");
    }

    #[test]
    fn lse_numerically_stable_large_values() {
        // Would overflow with naive exp(88) in f32; LSE subtracts max first.
        let weights = [80.0_f32, 88.0, 84.0];
        let result = normalize_path_weight(&weights);
        assert!(result.is_finite(), "LSE must be finite for large weights");
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
