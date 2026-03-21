//! Shared primitive types for CSRRE.
//! Everything here is Copy-sized so it can cross module boundaries cheaply.

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

/// MTLG modal type: mode ⊗ category ⊗ arity.
/// Directionality (/ vs \) is encoded in the signed arity: positive = right-arg, negative = left-arg.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModalType {
    pub mode:     ModalMode,
    pub category: TypeCategory,
    /// Number of remaining arguments (0 = saturated).
    pub arity:    u8,
    /// True = functor takes argument to the right (◇/), False = to the left (◇\).
    pub rightward: bool,
}

impl ModalType {
    pub fn atom(mode: ModalMode, category: TypeCategory) -> Self {
        Self { mode, category, arity: 0, rightward: true }
    }

    pub fn functor(mode: ModalMode, category: TypeCategory, arity: u8, rightward: bool) -> Self {
        Self { mode, category, arity, rightward }
    }

    /// Apply one argument; returns the result type after consuming one slot.
    pub fn apply(self) -> Option<Self> {
        if self.arity == 0 { return None; }
        Some(Self { arity: self.arity - 1, ..self })
    }

    /// Two types are compatible for composition if modes agree and result arity is consistent.
    pub fn compatible_with(self, arg: ModalType) -> bool {
        self.mode == arg.mode && self.arity > 0
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

/// An STO situation: typed partial world. Carries the TRD assignment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Situation {
    pub id:     u64,
    pub label:  String,
    pub trd:    Option<TRDId>,
    /// Assumption bit assigned by the ATMS for this situation.
    pub env_bit: u8,
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
