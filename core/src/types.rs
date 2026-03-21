//! Shared primitive types for CSRRE.
//! Everything here is Copy-sized so it can cross module boundaries cheaply.

use std::collections::HashSet;
use serde::{Deserialize, Serialize};

// ─── Identifier types ────────────────────────────────────────────────────────

pub type NodeId    = u64;
pub type EdgeId    = u64;
pub type ContextId = u64;
pub type TRDId     = u32;
/// ATMS environment: bitvector over assumption bits (up to 64 concurrent situations).
pub type Env       = u64;
pub type InfonId   = u64;

// ─── Modal type system (MTLG) ────────────────────────────────────────────────

/// Three modal modes from MTLG: ◇ (primary/linear), □ (shared/contraction), ◊ (discontinuous).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalMode {
    Diamond,  // ◇  primary composition — tree-forming, linear resource use
    Box,      // □  shared composition  — contraction permitted, DAG-forming
    Lozenge,  // ◊  discontinuous       — displacement calculus, scope/long-range
}

/// Argument directionality in the type-logical grammar.
///
/// In MTLG, `A/B` (Right) means the functor seeks its B argument to the right.
/// `A\B` (Left) means the functor seeks its B argument to the left.
/// These are distinct type constructors with distinct sequent calculus rules
/// (Moot & Retoré 2012 §2). A boolean is insufficient: it hides the semantic
/// content of the distinction from any reader of the type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    Right,  // /  — functor seeks argument to the right
    Left,   // \  — functor seeks argument to the left
}

/// UCCA-grounded semantic categories (typologically stable per BLT universals).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeCategory {
    Scene,
    Process,
    State,
    Participant,
    Adverbial,
    Connector,
    Ground,
}

/// MTLG modal type: mode ⊗ category ⊗ arity ⊗ direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModalType {
    pub mode:      ModalMode,
    pub category:  TypeCategory,
    /// Number of remaining unsaturated arguments (0 = fully saturated atom).
    pub arity:     u8,
    /// Which side the next argument must come from.
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
    ///
    /// `arg_is_right` must reflect where the argument sits relative to the functor
    /// in the actual derivation tree. A right-seeking functor (`Direction::Right`)
    /// requires `arg_is_right == true`; a left-seeking functor requires false.
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
    fn default() -> Self {
        Self::atom(ModalMode::Diamond, TypeCategory::Scene)
    }
}

// ─── Attribution quality ─────────────────────────────────────────────────────

/// Three-point quality signal from dissolved TR evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Quality {
    Good = 1,   // 1.0
    Partial = 2, // 0.5
    Bad = 0,    // 0.0
}

impl Quality {
    pub fn as_f64(self) -> f64 {
        match self {
            Quality::Good    => 1.0,
            Quality::Partial => 0.5,
            Quality::Bad     => 0.0,
        }
    }
}

// ─── Situation / STO ─────────────────────────────────────────────────────────

/// An STO situation: typed partial world.
///
/// `active_nodes` is the set of ARG NodeIds anchored in this situation.
/// An infon σ is supported by s only if all of σ's argument roles are filled
/// by nodes in `active_nodes` and σ's polarity is positive (Barwise & Perry 1983).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Situation {
    pub id:           u64,
    pub label:        String,
    pub trd:          Option<TRDId>,
    /// Assumption bit assigned by the ATMS for this situation.
    pub env_bit:      u8,
    /// NodeIds of ARG nodes anchored in this situation (fills argument roles).
    pub active_nodes: HashSet<NodeId>,
}

impl Situation {
    pub fn new(id: u64, label: impl Into<String>, env_bit: u8) -> Self {
        Self {
            id,
            label:        label.into(),
            trd:          None,
            env_bit,
            active_nodes: HashSet::new(),
        }
    }

    pub fn anchor(&mut self, node_id: NodeId) {
        self.active_nodes.insert(node_id);
    }
}

// ─── Infon ────────────────────────────────────────────────────────────────────

/// Infon σ: a typed proposition supported by a situation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Infon {
    pub id:       InfonId,
    pub relation: String,
    pub args:     Vec<NodeId>,
    pub polarity: bool,
}

// ─── Graphica cache key ───────────────────────────────────────────────────────

/// Canonical memoization key: (structural hash, ATMS environment, modal profile).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub hash:         u64,
    pub env:          Env,
    pub modal_mode:   ModalMode,
    pub modal_cat:    TypeCategory,
}
