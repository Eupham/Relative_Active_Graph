//! Symbolica-style algebraic normalization for modal lifting rules.
//! Handles the algebraic component of context shifts (unit conversion, scaling).
//! The MTLG module handles the type component; this module handles the numeric algebra.
//!
//! Symbolica (symbolica.io) is a CAS; here we implement the subset needed:
//! rational arithmetic + polynomial normalization over named variables.

use std::collections::HashMap;

/// A simple algebraic expression over named variables with rational coefficients.
#[derive(Clone, Debug, PartialEq)]
pub enum AlgExpr {
    Num(f64),
    Var(String),
    Add(Box<AlgExpr>, Box<AlgExpr>),
    Mul(Box<AlgExpr>, Box<AlgExpr>),
    Scale(f64, Box<AlgExpr>),
}

impl AlgExpr {
    /// Evaluate by substituting variables from the binding map.
    pub fn eval(&self, bindings: &HashMap<String, f64>) -> Option<f64> {
        match self {
            AlgExpr::Num(n)          => Some(*n),
            AlgExpr::Var(v)          => bindings.get(v).copied(),
            AlgExpr::Add(a, b)       => Some(a.eval(bindings)? + b.eval(bindings)?),
            AlgExpr::Mul(a, b)       => Some(a.eval(bindings)? * b.eval(bindings)?),
            AlgExpr::Scale(s, e)     => Some(s * e.eval(bindings)?),
        }
    }

    /// Normalize: collect like terms and simplify constants.
    pub fn normalize(self) -> Self {
        match self {
            AlgExpr::Add(a, b) => {
                let a = a.normalize();
                let b = b.normalize();
                match (&a, &b) {
                    (AlgExpr::Num(x), AlgExpr::Num(y)) => AlgExpr::Num(x + y),
                    (AlgExpr::Num(0.0), _) => b,
                    (_, AlgExpr::Num(0.0)) => a,
                    _ => AlgExpr::Add(Box::new(a), Box::new(b)),
                }
            }
            AlgExpr::Mul(a, b) => {
                let a = a.normalize();
                let b = b.normalize();
                match (&a, &b) {
                    (AlgExpr::Num(x), AlgExpr::Num(y)) => AlgExpr::Num(x * y),
                    (AlgExpr::Num(1.0), _) => b,
                    (_, AlgExpr::Num(1.0)) => a,
                    (AlgExpr::Num(0.0), _) | (_, AlgExpr::Num(0.0)) => AlgExpr::Num(0.0),
                    _ => AlgExpr::Mul(Box::new(a), Box::new(b)),
                }
            }
            AlgExpr::Scale(s, e) => {
                let e = e.normalize();
                if s == 1.0 { return e; }
                if s == 0.0 { return AlgExpr::Num(0.0); }
                if let AlgExpr::Num(n) = &e { return AlgExpr::Num(s * n); }
                AlgExpr::Scale(s, Box::new(e))
            }
            other => other,
        }
    }
}

/// A named algebraic transformation (e.g., unit conversion for modal lifting).
#[derive(Clone, Debug)]
pub struct AlgTransform {
    pub name:       String,
    /// The expression to apply. Input variable is "x".
    pub expression: AlgExpr,
}

impl AlgTransform {
    pub fn scale(name: impl Into<String>, factor: f64) -> Self {
        Self {
            name:       name.into(),
            expression: AlgExpr::Scale(factor, Box::new(AlgExpr::Var("x".into()))),
        }
    }

    pub fn apply(&self, value: f64) -> Option<f64> {
        let mut b = HashMap::new();
        b.insert("x".into(), value);
        self.expression.eval(&b)
    }
}

/// Pre-e-graph algebraic normalization: normalize all AlgExpr in active nodes.
/// Called before equality saturation.
pub struct AlgebraicNormalizer {
    transforms: HashMap<String, AlgTransform>,
}

impl AlgebraicNormalizer {
    pub fn new() -> Self {
        Self { transforms: HashMap::new() }
    }

    pub fn register(&mut self, t: AlgTransform) {
        self.transforms.insert(t.name.clone(), t);
    }

    /// Normalize a named expression with a given input value.
    pub fn normalize(&self, transform_name: &str, value: f64) -> Option<f64> {
        self.transforms.get(transform_name)?.apply(value)
    }

    /// Apply all registered transforms whose names appear in the node surface.
    pub fn apply_to_surface(&self, surface: &str, value: f64) -> f64 {
        for (name, transform) in &self.transforms {
            if surface.contains(name.as_str()) {
                if let Some(result) = transform.apply(value) {
                    return result;
                }
            }
        }
        value
    }
}

impl Default for AlgebraicNormalizer {
    fn default() -> Self { Self::new() }
}

// ── Path weight normalization ─────────────────────────────────────────────────

/// Harmonic mean of constituent edge weights for a shortcut edge.
/// Penalizes weak links: a path [1.0, 1.0, 0.1] returns ~0.25.
pub fn normalize_path_weight(weights: &[f32]) -> f32 {
    if weights.is_empty() { return 0.0; }
    let n = weights.len() as f32;
    let sum_recip: f32 = weights.iter()
        .map(|&w| if w > 1e-9 { 1.0 / w } else { f32::MAX })
        .sum();
    if sum_recip == 0.0 || sum_recip.is_infinite() { return 0.0; }
    n / sum_recip
}

/// Apply a named algebraic transform to `weight` via the normalizer,
/// or return `weight` unchanged if no matching transform is registered.
/// Called by `lifting.rs` when a ModalLiftingRule has algebraic_transform set.
pub fn apply_named_transform(
    normalizer: &AlgebraicNormalizer,
    name:       &str,
    weight:     f32,
) -> f32 {
    normalizer.normalize(name, weight as f64).map(|v| v as f32).unwrap_or(weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_transform() {
        let t = AlgTransform::scale("usd-to-eur", 0.92);
        assert!((t.apply(100.0).unwrap() - 92.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_adds_constants() {
        let expr = AlgExpr::Add(Box::new(AlgExpr::Num(3.0)), Box::new(AlgExpr::Num(4.0)));
        assert_eq!(expr.normalize(), AlgExpr::Num(7.0));
    }

    #[test]
    fn normalize_mul_by_zero() {
        let expr = AlgExpr::Mul(Box::new(AlgExpr::Num(0.0)), Box::new(AlgExpr::Var("x".into())));
        assert_eq!(expr.normalize(), AlgExpr::Num(0.0));
    }

    #[test]
    fn harmonic_mean_penalizes_weak_link() {
        let w = normalize_path_weight(&[1.0, 1.0, 0.1]);
        assert!(w < 0.3, "got {w}");
    }
    #[test]
    fn harmonic_mean_uniform() {
        let w = normalize_path_weight(&[0.8, 0.8, 0.8]);
        assert!((w - 0.8).abs() < 1e-5, "got {w}");
    }
    #[test]
    fn empty_path_is_zero() {
        assert_eq!(normalize_path_weight(&[]), 0.0);
    }
}
