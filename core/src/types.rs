//! Shared primitive types for CSRRE.

use std::collections::HashSet;
use serde::{Deserialize, Serialize};

// ─── Identifier types ────────────────────────────────────────────────────────

pub type NodeId    = u64;
pub type EdgeId    = u64;
pub type ContextId = u64;
pub type TRDId     = u32;
pub type Env       = u64;
pub type InfonId   = u64;

// ─── Modal mode (mathematical constant from MTLG) ─────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalMode {
    Diamond,  // ◇  primary composition — tree-forming, linear
    Box,      // □  shared composition  — contraction, DAG-forming
    Lozenge,  // ◊  discontinuous       — displacement calculus
}

// ─── Direction (mathematical constant from type-logical grammar) ──────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    Right,  // /  — functor seeks argument to the right
    Left,   // \  — functor seeks argument to the left
}

// ─── TypeCategory: runtime-discovered cluster ID ─────────────────────────────

/// Semantic category ID discovered by CategoryInducer bootstrap.
/// 0 = DEFAULT (unassigned). All other values are assigned at bootstrap time.
/// No named variants — the system must discover and designate category concepts
/// from structural evidence, not from pre-specified labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TypeCategory(pub u32);

impl TypeCategory {
    pub const DEFAULT: TypeCategory = TypeCategory(0);
    pub fn id(self) -> u32 { self.0 }
    pub fn is_assigned(self) -> bool { self.0 != 0 }
}

impl std::fmt::Display for TypeCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cat:{}", self.0)
    }
}

// ─── MTLG modal type ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModalType {
    pub mode:      ModalMode,
    pub category:  TypeCategory,
    pub arity:     u8,
    pub direction: Direction,
}

impl ModalType {
    pub fn atom(mode: ModalMode, category: TypeCategory) -> Self {
        Self { mode, category, arity: 0, direction: Direction::Right }
    }

    pub fn functor(mode: ModalMode, category: TypeCategory, arity: u8, dir: Direction) -> Self {
        Self { mode, category, arity, direction: dir }
    }

    /// Consume one argument slot; returns the result type or None if saturated.
    pub fn apply(self) -> Option<Self> {
        if self.arity == 0 { return None; }
        Some(Self { arity: self.arity - 1, ..self })
    }

    /// True if this functor can combine with an argument type at the given position.
    pub fn compatible_with(self, arg: ModalType, arg_is_right: bool) -> bool {
        self.mode == arg.mode
            && self.arity > 0
            && match self.direction {
                Direction::Right => arg_is_right,
                Direction::Left  => !arg_is_right,
            }
    }
}

impl Default for ModalType {
    fn default() -> Self { Self::atom(ModalMode::Diamond, TypeCategory::DEFAULT) }
}

// ─── Continuous quality signal ────────────────────────────────────────────────

/// Quality signal in [0.0, 1.0] from TR dissolution or CE training comparison.
/// Continuous to support per-position quality from long-output training.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Quality(pub f32);

impl Quality {
    pub const GOOD:    Quality = Quality(1.0);
    pub const PARTIAL: Quality = Quality(0.5);
    pub const BAD:     Quality = Quality(0.0);

    pub fn new(v: f32) -> Self { Quality(v.clamp(0.0, 1.0)) }
    pub fn as_f32(self) -> f32  { self.0 }
    pub fn as_f64(self) -> f64  { self.0 as f64 }

    /// Cross-entropy quality signal.
    /// `p_predicted`: softmax probability of the token that was generated.
    /// `was_correct`: whether that token matched ground truth.
    pub fn from_ce(p_predicted: f32, was_correct: bool) -> Self {
        if was_correct {
            Quality::new(1.0 - p_predicted)  // CE gradient ∂L/∂z_correct = p - 1 (inverted)
        } else {
            Quality::BAD
        }
    }

    /// Token overlap for long-output position quality.
    pub fn token_overlap(generated: &str, expected: &str) -> Quality {
        let gen: HashSet<&str> = generated.split_whitespace().collect();
        let exp: Vec<&str>     = expected.split_whitespace().collect();
        if exp.is_empty() { return Quality::BAD; }
        let matched = exp.iter().filter(|t| gen.contains(*t)).count();
        Quality::new(matched as f32 / exp.len() as f32)
    }
}

impl Default for Quality { fn default() -> Self { Quality::BAD } }

impl std::fmt::Display for Quality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

// ─── Situation / STO ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Situation {
    pub id:           u64,
    pub label:        String,
    pub trd:          Option<TRDId>,
    pub env_bit:      u8,
    pub active_nodes: HashSet<NodeId>,
}

impl Situation {
    pub fn new(id: u64, label: impl Into<String>, env_bit: u8) -> Self {
        Self { id, label: label.into(), trd: None, env_bit, active_nodes: HashSet::new() }
    }
    pub fn anchor(&mut self, node_id: NodeId) { self.active_nodes.insert(node_id); }
}

// ─── Infon ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Infon {
    pub id:       InfonId,
    pub relation: String,
    pub args:     Vec<NodeId>,
    pub polarity: bool,
}

// ─── Cache key ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub hash:       u64,
    pub env:        Env,
    pub modal_mode: ModalMode,
    pub modal_cat:  TypeCategory,
}
