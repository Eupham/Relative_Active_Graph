import urllib.request
import json
import sys

# Minimal viable snippet representation of the types used in scm.rs to check for syntax errors
rust_code = """
use std::collections::{HashMap, HashSet};

type NodeId = u64;

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
}

pub struct StructuralEq {
    pub variable:  NodeId,
    pub parents:   Vec<NodeId>,
    pub coeffs:    Vec<f64>,
    pub intercept: f64,
    pub noise:     Distribution,
}

impl StructuralEq {
    pub fn expected_value_given_evidence(&self, parent_values: &HashMap<NodeId, f64>) -> f64 {
        let linear: f64 = self.parents.iter().zip(&self.coeffs)
            .filter_map(|(&p, &c)| parent_values.get(&p).map(|&v| c * v))
            .sum::<f64>();

        let exog_ev = self.noise.expected_value();
        linear + self.intercept + exog_ev
    }
}

fn main() {
    println!("Compiled successfully!");
}
"""

req = urllib.request.Request(
    'https://play.rust-lang.org/execute',
    data=json.dumps({
        'channel': 'stable',
        'mode': 'debug',
        'edition': '2021',
        'crateType': 'bin',
        'code': rust_code
    }).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    response = urllib.request.urlopen(req)
    result = json.loads(response.read())
    print("STDOUT:", result.get('stdout', ''))
    print("STDERR:", result.get('stderr', ''))
except Exception as e:
    print("Error contacting Rust playground:", e)
