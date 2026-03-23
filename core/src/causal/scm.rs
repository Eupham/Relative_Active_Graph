//! Structural Causal Model (SCM) over the ARG.
//! Implements Pearl's do-calculus framework: causal edges, interventions, counterfactuals.
//! Language: V = observed variables (ARG nodes), U = exogenous noise, F = structural equations.

use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use crate::types::{NodeId, EdgeId, TRDId};

/// Discrete exogenous perturbation: models structural noise as a Poisson process (§17).
/// `perturbation_rate` (λ) is the expected count of discrete signal-drop events
/// per SCM computation cycle. λ = 0.0 → fully deterministic.
/// Same λ is shared with MetaGrammarEngine::absorb_rule for rule perturbation (§12).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExogVar {
    pub id:                u64,
    /// Poisson λ — replaces std_dev. Governs discrete structural perturbation rate.
    pub perturbation_rate: f64,
}

/// A structural equation: V_i = f_i(Pa(V_i), U_i).
/// Represented as linear function for tractability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuralEq {
    pub variable:  NodeId,
    pub parents:   Vec<NodeId>,
    /// Coefficients for each parent (linear SCM).
    pub coeffs:    Vec<f64>,
    pub intercept: f64,
    pub noise:     ExogVar,
}

impl StructuralEq {
    /// Evaluate using Poisson discrete perturbation (§17).
    /// n_events ~ Poisson(λ) signal-drop events attenuate the linear sum.
    /// Callers requiring deterministic baseline use ExogVar { perturbation_rate: 0.0 }.
    pub fn evaluate(&self, parent_values: &HashMap<NodeId, f64>, rng: &mut impl rand::Rng) -> f64 {
        let linear: f64 = self.parents.iter().zip(&self.coeffs)
            .filter_map(|(&p, &c)| parent_values.get(&p).map(|&v| c * v))
            .sum();

        let n_events: f64 = if self.noise.perturbation_rate > 0.0 {
            rng.sample(rand_distr::Poisson::new(self.noise.perturbation_rate)
                .expect("valid Poisson parameter"))
        } else {
            0.0
        };

        let attenuation = n_events / self.parents.len().max(1) as f64;
        (linear + self.intercept) * (1.0_f64 - attenuation).max(0.0_f64)
    }

    /// Deterministic evaluate (zero noise). Used for point estimates and tests.
    pub fn evaluate_deterministic(&self, parent_values: &HashMap<NodeId, f64>) -> f64 {
        let linear: f64 = self.parents.iter().zip(&self.coeffs)
            .filter_map(|(&p, &c)| parent_values.get(&p).map(|&v| c * v))
            .sum();
        linear + self.intercept
    }
}

/// SCM over the ARG: V = {NodeId}, E = causal edges, F = structural equations.
#[derive(Clone, Serialize, Deserialize)]
pub struct Scm {
    /// Causal adjacency: parent → set of children.
    pub parents:   HashMap<NodeId, Vec<NodeId>>,
    pub equations: HashMap<NodeId, StructuralEq>,
    /// Observed values in the current context.
    pub values:    HashMap<NodeId, f64>,
}

impl Scm {
    pub fn new() -> Self {
        Self {
            parents:   HashMap::new(),
            equations: HashMap::new(),
            values:    HashMap::new(),
        }
    }

    pub fn add_equation(&mut self, eq: StructuralEq) {
        for &parent in &eq.parents {
            self.parents.entry(parent).or_default().push(eq.variable);
        }
        self.equations.insert(eq.variable, eq);
    }

    pub fn observe(&mut self, var: NodeId, value: f64) {
        self.values.insert(var, value);
    }

    /// Compute the value of `var` under current observations (deterministic, recursive).
    pub fn compute(&self, var: NodeId) -> Option<f64> {
        if let Some(&v) = self.values.get(&var) { return Some(v); }
        let eq = self.equations.get(&var)?;
        let mut parent_vals = HashMap::new();
        for &p in &eq.parents {
            parent_vals.insert(p, self.compute(p)?);
        }
        Some(eq.evaluate_deterministic(&parent_vals))
    }

    /// do(X=val): hard intervention — remove incoming edges to X, set X=val.
    /// Returns a new SCM with X's equation replaced by a constant.
    pub fn intervene(&self, target: NodeId, value: f64) -> Scm {
        let mut new_scm = Scm {
            parents:   self.parents.clone(),
            equations: self.equations.clone(),
            values:    self.values.clone(),
        };
        // Override with constant equation (no parents).
        new_scm.equations.insert(target, StructuralEq {
            variable:  target,
            parents:   vec![],
            coeffs:    vec![],
            intercept: value,
            noise:     ExogVar { id: 0, perturbation_rate: 0.0 },
        });
        new_scm.values.insert(target, value);
        new_scm
    }

    /// Returns a new Scm in which the structural equation for `variable` has the
    /// coefficient for `parent` set to zero, severing the edge parent → variable.
    /// Pearl do-calculus: do(X=absent) ≡ setting all β_k = 0 for parent X.
    /// The original Scm is preserved unchanged (immutable intervention, §3).
    pub fn remove_parent(&self, variable: NodeId, parent: NodeId) -> Scm {
        let mut new_scm = self.clone();
        if let Some(eq) = new_scm.equations.get_mut(&variable) {
            if let Some(pos) = eq.parents.iter().position(|&p| p == parent) {
                eq.coeffs[pos] = 0.0;
            }
        }
        new_scm
    }

    /// Topological sort of SCM variables (Kahn's algorithm).
    pub fn topological_order(&self) -> Vec<NodeId> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut children:  HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for (&var, eq) in &self.equations {
            in_degree.entry(var).or_insert(0);
            for &parent in &eq.parents {
                in_degree.entry(parent).or_insert(0);
                *in_degree.entry(var).or_insert(0) += 1;
                children.entry(parent).or_default().push(var);
            }
        }

        let mut queue: Vec<NodeId> = in_degree.iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&v, _)| v)
            .collect();
        let mut order = Vec::new();

        while let Some(v) = queue.pop() {
            order.push(v);
            if let Some(ch) = children.get(&v) {
                for &c in ch {
                    let d = in_degree.get_mut(&c).unwrap();
                    *d -= 1;
                    if *d == 0 { queue.push(c); }
                }
            }
        }
        order
    }
}

impl Default for Scm { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_linear_chain() {
        let mut scm = Scm::new();
        // X → Y: Y = 2X + 1
        scm.observe(1, 3.0); // X = 3
        scm.add_equation(StructuralEq {
            variable:  2,
            parents:   vec![1],
            coeffs:    vec![2.0],
            intercept: 1.0,
            noise:     ExogVar { id: 0, perturbation_rate: 0.0 },
        });
        assert_eq!(scm.compute(2), Some(7.0)); // 2*3+1=7
    }

    #[test]
    fn intervention_overrides_equation() {
        let mut scm = Scm::new();
        scm.observe(1, 3.0);
        scm.add_equation(StructuralEq {
            variable:  2, parents: vec![1], coeffs: vec![2.0], intercept: 1.0,
            noise: ExogVar { id: 0, perturbation_rate: 0.0 },
        });
        let intervened = scm.intervene(2, 99.0);
        assert_eq!(intervened.compute(2), Some(99.0));
    }
}
