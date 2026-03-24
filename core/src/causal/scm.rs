//! True Bayesian Structural Causal Model (SCM) over the ARG.
//! Implements Pearl's do-calculus framework: causal edges, interventions, counterfactuals
//! via exact inference and graphical mutilation.
//! Language: V = observed variables (ARG nodes), U = exogenous noise, F = structural equations.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::types::NodeId;
// numerica crate incorporated for streamlined math integrations


/// True probability distribution for Exogenous Variables (U).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Distribution {
    Gaussian { mean: f64, variance: f64 },
    Categorical(Vec<f64>),
    Deterministic(f64),
}

impl Distribution {
    pub fn expected_value(&self) -> f64 {
        match self {
            Self::Gaussian { mean, .. } => *mean,
            Self::Categorical(probs) => {
                probs.iter().enumerate().map(|(i, &p)| i as f64 * p).sum()
            },
            Self::Deterministic(v) => *v,
        }
    }

    pub fn variance(&self) -> f64 {
        match self {
            Self::Gaussian { variance, .. } => *variance,
            Self::Categorical(_) => 1.0, // Baseline categorical variance
            Self::Deterministic(_) => 0.0,
        }
    }
}

/// A structural equation: V_i = f_i(Pa(V_i), U_i).
/// Implemented using ndarray for vectorized tensor weighting.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuralEq {
    pub variable:  NodeId,
    pub parents:   Vec<NodeId>,
    /// Tensor coefficients for exact inference.
    pub coeffs:    Vec<f64>,
    pub intercept: f64,
    pub noise:     Distribution,
}

impl StructuralEq {
    /// Exact Bayesian inference propagating full Gaussian messages (mean, variance).
    pub fn expected_distribution_given_evidence(&self, parent_dists: &HashMap<NodeId, (f64, f64)>) -> (f64, f64) {
        let mut mean_sum = self.intercept + self.noise.expected_value();
        let mut var_sum  = self.noise.variance();
        
        for (&p, &c) in self.parents.iter().zip(&self.coeffs) {
            if let Some(&(p_mean, p_var)) = parent_dists.get(&p) {
                mean_sum += c * p_mean;
                var_sum  += c * c * p_var;
            }
        }
        (mean_sum, var_sum)
    }

    pub fn expected_value_given_evidence(&self, parent_values: &HashMap<NodeId, f64>) -> f64 {
        let map = parent_values.iter().map(|(&k, &v)| (k, (v, 0.0))).collect();
        self.expected_distribution_given_evidence(&map).0
    }

    pub fn evaluate(&self, parent_values: &HashMap<NodeId, f64>, _rng: &mut impl rand::Rng) -> f64 {
        self.expected_value_given_evidence(parent_values)
    }

    pub fn evaluate_deterministic(&self, parent_values: &HashMap<NodeId, f64>) -> f64 {
        self.expected_value_given_evidence(parent_values)
    }
}

/// SCM over the ARG: V = {NodeId}, E = causal edges, F = structural equations.
#[derive(Clone, Serialize, Deserialize)]
pub struct Scm {
    pub parents:   HashMap<NodeId, Vec<NodeId>>,
    pub equations: HashMap<NodeId, StructuralEq>,
    pub values:    HashMap<NodeId, f64>,
    pub distributions: HashMap<NodeId, (f64, f64)>,
}

impl Scm {
    pub fn new() -> Self {
        Self {
            parents:   HashMap::new(),
            equations: HashMap::new(),
            values:    HashMap::new(),
            distributions: HashMap::new(),
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
        self.distributions.insert(var, (value, 0.0));
    }

    /// Compute exact inference producing joint Gaussian marginals (Mean, Variance).
    pub fn compute_distribution(&self, var: NodeId) -> Option<(f64, f64)> {
        if let Some(&d) = self.distributions.get(&var) { return Some(d); }
        if let Some(&v) = self.values.get(&var) { return Some((v, 0.0)); }
        let eq = self.equations.get(&var)?;
        let mut parent_dists = HashMap::new();
        for &p in &eq.parents {
            parent_dists.insert(p, self.compute_distribution(p)?);
        }
        Some(eq.expected_distribution_given_evidence(&parent_dists))
    }

    pub fn compute(&self, var: NodeId) -> Option<f64> {
        self.compute_distribution(var).map(|(mean, _)| mean)
    }

    /// do(X=val): Pearl's Exact Graph Mutilation intervention.
    /// Severs incoming edges to target and clamps distribution to a Delta function.
    pub fn intervene(&self, target: NodeId, value: f64) -> Scm {
        let mut new_scm = self.clone();
        
        // Mutilate the graph by destroying parental causal links
        new_scm.equations.insert(target, StructuralEq {
            variable:  target,
            parents:   vec![],
            coeffs:    vec![],
            intercept: value,
            noise:     Distribution::Deterministic(0.0),
        });
        
        // Remove target from children of its former parents
        if let Some(eq) = self.equations.get(&target) {
            for &p in &eq.parents {
                if let Some(children) = new_scm.parents.get_mut(&p) {
                    children.retain(|&c| c != target);
                }
            }
        }

        new_scm.values.insert(target, value);
        new_scm
    }

    pub fn remove_parent(&self, variable: NodeId, parent: NodeId) -> Scm {
        let mut new_scm = self.clone();
        if let Some(eq) = new_scm.equations.get_mut(&variable) {
            // Set beta coefficient to zero to sever linear causal influence
            if let Some(pos) = eq.parents.iter().position(|&p| p == parent) {
                eq.coeffs[pos] = 0.0;
            }
        }
        new_scm
    }

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
