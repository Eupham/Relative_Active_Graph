//! ARG edge: typed, modal, ATMS-justified.

use serde::{Deserialize, Serialize};
use crate::types::{NodeId, EdgeId, Env, ModalMode};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    Composition,  // MTLG functor-argument application
    Dependency,   // UD-style syntactic dependency
    Causal,       // SCM causal edge
    Equivalence,  // equality saturation equivalence
    Contextual,   // cross-context lifting
    Rule,         // rule application edge
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgEdge {
    pub id:        EdgeId,
    pub src:       NodeId,
    pub dst:       NodeId,
    pub edge_type: EdgeType,
    pub modal_mode: ModalMode,
    /// Edge weight updated by attribution trace on TR dissolution.
    pub weight:    f32,
    /// ATMS justification: antecedent NodeIds supporting this edge.
    pub atms_justification: Vec<NodeId>,
    /// Canonical ID for Graphica memoization: points to an equivalence class.
    pub canonical_id: Option<u64>,
}

impl ArgEdge {
    pub fn new(
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: EdgeType,
        modal_mode: ModalMode,
    ) -> Self {
        Self {
            id,
            src,
            dst,
            edge_type,
            modal_mode,
            weight: 0.5,
            atms_justification: vec![src, dst],
            canonical_id: None,
        }
    }

    /// True if this edge should be included in G(s) given the edge threshold `theta_rho`.
    pub fn is_active(&self, theta_rho: f64) -> bool {
        self.weight as f64 > theta_rho
    }

    /// Compute the environment that supports this edge: union of antecedent labels.
    pub fn required_env(&self, label_of: impl Fn(NodeId) -> Option<Env>) -> Option<Env> {
        let mut env = 0u64;
        for &nid in &self.atms_justification {
            env |= label_of(nid)?;
        }
        Some(env)
    }

    /// Apply attribution delta to weight (clamped to [0,1]).
    pub fn apply_delta(&mut self, delta: f32) {
        self.weight = (self.weight + delta).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModalMode;

    #[test]
    fn edge_active_above_threshold() {
        let mut e = ArgEdge::new(1, 10, 20, EdgeType::Composition, ModalMode::Diamond);
        e.weight = 0.7;
        assert!(e.is_active(0.5));
        assert!(!e.is_active(0.8));
    }

    #[test]
    fn delta_clamped() {
        let mut e = ArgEdge::new(1, 10, 20, EdgeType::Composition, ModalMode::Diamond);
        e.weight = 0.9;
        e.apply_delta(0.5);
        assert_eq!(e.weight, 1.0);
        e.apply_delta(-2.0);
        assert_eq!(e.weight, 0.0);
    }
}
