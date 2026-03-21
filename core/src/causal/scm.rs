//! Structural Causal Model (SCM) over the ARG.
//! Implements Pearl's do-calculus framework: causal edges, interventions, counterfactuals.
//! Language: V = observed variables (ARG nodes), U = exogenous noise, F = structural equations.

use std::collections::{HashMap, HashSet};
use crate::types::{NodeId, EdgeId, TRDId};

/// An exogenous noise variable.
#[derive(Clone, Debug)]
pub struct ExogVar {
    pub id:       u64,
    /// Standard deviation of this noise variable's distribution.
    pub std_dev:  f64,
}

/// A structural equation: V_i = f_i(Pa(V_i), U_i).
/// Represented as linear function for tractability.
#[derive(Clone, Debug)]
pub struct StructuralEq {
    pub variable:  NodeId,
    pub parents:   Vec<NodeId>,
    /// Coefficients for each parent (linear SCM).
    pub coeffs:    Vec<f64>,
    pub intercept: f64,
    pub noise:     ExogVar,
}

impl StructuralEq {
    /// Evaluate: f(parent_values) = sum(coeff_i * parent_i) + intercept + noise.
    pub fn evaluate(&self, parent_values: &HashMap<NodeId, f64>, noise_value: f64) -> f64 {
        let linear: f64 = self.parents.iter().zip(&self.coeffs)
            .filter_map(|(&p, &c)| parent_values.get(&p).map(|&v| c * v))
            .sum();
        linear + self.intercept + noise_value * self.noise.std_dev
    }
}

/// SCM over the ARG: V = {NodeId}, E = causal edges, F = structural equations.
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

    /// Compute the value of `var` under current observations (recursive, topological).
    pub fn compute(&self, var: NodeId) -> Option<f64> {
        if let Some(&v) = self.values.get(&var) { return Some(v); }
        let eq = self.equations.get(&var)?;
        let mut parent_vals = HashMap::new();
        for &p in &eq.parents {
            parent_vals.insert(p, self.compute(p)?);
        }
        Some(eq.evaluate(&parent_vals, 0.0)) // deterministic (noise=0 for point estimate)
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
            noise:     ExogVar { id: 0, std_dev: 0.0 },
        });
        new_scm.values.insert(target, value);
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
            noise:     ExogVar { id: 0, std_dev: 0.0 },
        });
        assert_eq!(scm.compute(2), Some(7.0)); // 2*3+1=7
    }

    #[test]
    fn intervention_overrides_equation() {
        let mut scm = Scm::new();
        scm.observe(1, 3.0);
        scm.add_equation(StructuralEq {
            variable:  2, parents: vec![1], coeffs: vec![2.0], intercept: 1.0,
            noise: ExogVar { id: 0, std_dev: 0.0 },
        });
        let intervened = scm.intervene(2, 99.0);
        assert_eq!(intervened.compute(2), Some(99.0));
    }
}
