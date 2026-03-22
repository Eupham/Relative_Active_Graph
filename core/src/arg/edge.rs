//! ARG edge. EdgeClass(u32) replaces the named EdgeType enum.
//! ModalMode is retained as a mathematical constant.

use serde::{Deserialize, Serialize};
use crate::types::{NodeId, EdgeId, Env, ModalMode};

/// Discovered edge class ID from bootstrap clustering. 0 = DEFAULT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct EdgeClass(pub u32);

impl EdgeClass {
    pub const DEFAULT: EdgeClass = EdgeClass(0);
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgEdge {
    pub id:        EdgeId,
    pub src:       NodeId,
    pub dst:       NodeId,
    pub edge_class: EdgeClass,
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
        edge_class: EdgeClass,
        modal_mode: ModalMode,
    ) -> Self {
        Self {
            id, src, dst, edge_class, modal_mode,
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
        let mut e = ArgEdge::new(1, 10, 20, EdgeClass::DEFAULT, ModalMode::Diamond);
        e.weight = 0.7;
        assert!(e.is_active(0.5));
        assert!(!e.is_active(0.8));
    }

    #[test]
    fn delta_clamped() {
        let mut e = ArgEdge::new(1, 10, 20, EdgeClass::DEFAULT, ModalMode::Diamond);
        e.weight = 0.9;
        e.apply_delta(0.5);
        assert_eq!(e.weight, 1.0);
        e.apply_delta(-2.0);
        assert_eq!(e.weight, 0.0);
    }
}
