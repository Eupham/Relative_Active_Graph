//! ARG node. Structural role determined by position, arity, and attribution.
//! NodeClass(u32) replaces NodeType named enum — class IDs assigned at bootstrap.

use serde::{Deserialize, Serialize};
use crate::types::{NodeId, Env, ModalType, TRDId, InfonId};

/// Discovered node class ID from bootstrap clustering. 0 = unclassified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct NodeClass(pub u32);

impl NodeClass {
    pub const DEFAULT: NodeClass = NodeClass(0);
    pub fn is_assigned(self) -> bool { self.0 != 0 }
}

fn compute_depth(path_density: f32, attribution_score: f32, type_arity: u8, causal_parent_count: usize) -> f32 {
    let type_complexity = (type_arity as f32 + 1.0).ln();
    let causal_factor   = (causal_parent_count as f32 + 1.0).ln();
    (path_density * 0.4) + (attribution_score * 0.3) + (type_complexity * 0.2) + (causal_factor * 0.1)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgNode {
    pub id:                NodeId,
    pub node_class:        NodeClass,
    /// Language-agnostic surface bytes (UTF-8 string stored as bytes; None for abstract nodes).
    pub surface:           Option<Vec<u8>>,
    /// Activation infon: the (situation_id, infon_id) pair that triggers this node.
    pub activation_infon:  (u64, InfonId),
    /// Singleton ATMS label for base layer.
    pub atms_label:        Env,
    /// Locally inferred depth in G(s).
    pub depth:             f32,
    /// Attribution score accumulated from dissolved TR traces.
    pub attribution_score: f32,
    pub mtlg_type:         ModalType,
    pub trd_membership:    Vec<TRDId>,
    /// Sequential step at which this node was last activated (0 = never).
    /// Reserved for future recency-weighting; not yet used in computation.
    pub last_activated_step: u64,
    /// Number of times this node has been activated across all passages.
    /// Reserved for future recency-weighting; not yet used in computation.
    pub activation_count:    u32,
}

impl ArgNode {
    pub fn new(id: NodeId, node_class: NodeClass, mtlg_type: ModalType, activation_infon: (u64, InfonId)) -> Self {
        Self {
            id, node_class, surface: None, activation_infon,
            atms_label: 0, depth: 0.0, attribution_score: 0.5,
            mtlg_type, trd_membership: Vec::new(),
            last_activated_step: 0, activation_count: 0,
        }
    }

    pub fn with_surface(mut self, bytes: Vec<u8>) -> Self { self.surface = Some(bytes); self }
    pub fn with_label(mut self, env: Env) -> Self { self.atms_label = env; self }

    /// Recompute depth given current context stats.
    pub fn update_depth(&mut self, path_density: f32, causal_parent_count: usize) {
        self.depth = compute_depth(path_density, self.attribution_score, self.mtlg_type.arity, causal_parent_count);
    }

    /// True if this node is active in `active_env` given threshold `theta`.
    pub fn is_active(&self, active_env: Env, theta: f64) -> bool {
        use crate::atms::base::env::subsumes;
        subsumes(self.atms_label, active_env) && self.attribution_score as f64 > theta
    }

    pub fn surface_str(&self) -> Option<&str> {
        self.surface.as_deref().and_then(|b| std::str::from_utf8(b).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalMode, TypeCategory};

    fn make_node(id: NodeId) -> ArgNode {
        let mt = ModalType::atom(ModalMode::Diamond, TypeCategory::DEFAULT);
        ArgNode::new(id, NodeClass::DEFAULT, mt, (0, 0))
            .with_surface(b"test".to_vec())
            .with_label(0b11)
    }

    #[test]
    fn node_active_in_superset_env() {
        let n = make_node(1);
        assert!(n.is_active(0b111, 0.0));
    }

    #[test]
    fn node_inactive_below_threshold() {
        let mut n = make_node(1);
        n.attribution_score = 0.1;
        assert!(!n.is_active(0b111, 0.5));
    }
}
