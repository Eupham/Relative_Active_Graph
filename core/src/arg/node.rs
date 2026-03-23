//! ARG node. Structure field carries surface features for online category induction.

use serde::{Deserialize, Serialize};
use crate::types::{NodeId, Env, ModalType, TRDId, InfonId};
use crate::lcs::token_types::TokenStructure;

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
    pub surface:           Option<Vec<u8>>,
    pub activation_infon:  (u64, InfonId),
    pub atms_label:        Env,
    pub depth:             f32,
    pub attribution_score: f32,
    pub mtlg_type:         ModalType,
    pub trd_membership:    Vec<TRDId>,
    pub last_activated_step: u64,
    pub activation_count:    u32,
    /// Surface feature record for online category re-induction.
    /// None for abstract/synthetic nodes.
    #[serde(skip)]
    pub structure: Option<TokenStructure>,
}

impl ArgNode {
    pub fn new(id: NodeId, node_class: NodeClass, mtlg_type: ModalType, activation_infon: (u64, InfonId)) -> Self {
        Self {
            id, node_class, surface: None, activation_infon,
            atms_label: 0, depth: 0.0, attribution_score: 0.5,
            mtlg_type, trd_membership: Vec::new(),
            last_activated_step: 0, activation_count: 0,
            structure: None,
        }
    }

    pub fn with_surface(mut self, bytes: Vec<u8>) -> Self { self.surface = Some(bytes); self }
    pub fn with_label(mut self, env: Env) -> Self { self.atms_label = env; self }

    pub fn update_depth(&mut self, path_density: f32, causal_parent_count: usize) {
        self.depth = compute_depth(path_density, self.attribution_score, self.mtlg_type.arity, causal_parent_count);
    }

    pub fn is_active(&self, active_env: Env, theta: f64) -> bool {
        (self.atms_label & active_env) != 0 && self.attribution_score as f64 >= theta
    }

    pub fn surface_str(&self) -> Option<&str> {
        self.surface.as_deref().and_then(|b| std::str::from_utf8(b).ok())
    }
}
