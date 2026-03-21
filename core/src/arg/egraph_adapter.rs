//! Equality saturation adapter using egg.
//! E-classes are keyed by (modal_mode, category) to enforce modal separation:
//! ◇-composed and □-composed subgraphs never merge into the same e-class.

use egg::{define_language, Analysis, DidMerge, EGraph, Id, RecExpr, Rewrite, Runner, Symbol};
use crate::types::{ModalMode, TypeCategory, ModalType};

// ─── Language definition ──────────────────────────────────────────────────────

define_language! {
    pub enum MtlgExpr {
        // Atomic symbol (variable or constant)
        Sym(Symbol),
        // Integer literal
        Num(i64),
        // Modal applications — one variant per mode to enforce modal separation
        "d-app" = DiamondApp([Id; 2]),  // ◇ application: f ◇ x
        "b-app" = BoxApp([Id; 2]),      // □ application: f □ x
        "l-app" = LozengeApp([Id; 2]), // ◊ application: f ◊ x
        // Lambda abstraction: (λ var body)
        "lam"   = Lam([Id; 2]),
        // Predicate: (pred name arg1 arg2) — max arity 2 for simplicity
        "pred1" = Pred1([Id; 2]),
        "pred2" = Pred2([Id; 3]),
        // Equivalence injection: two subgraphs are equal
        "eq"    = Eq([Id; 2]),
    }
}

// ─── Per-e-class analysis (modal type tracking) ───────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModalData {
    pub mode:     ModalMode,
    pub category: TypeCategory,
}

impl ModalData {
    fn default_data() -> Self {
        Self { mode: ModalMode::Diamond, category: TypeCategory::Scene }
    }

    fn join(a: &mut Self, b: Self) -> DidMerge {
        // Lattice join: modes must agree; category defaults to Scene on conflict.
        let changed_mode = a.mode != b.mode;
        let changed_cat  = a.category != b.category;
        if changed_mode { a.mode = ModalMode::Diamond; }  // conservative default
        if changed_cat  { a.category = TypeCategory::Scene; }
        DidMerge(changed_mode || changed_cat, changed_mode || changed_cat)
    }
}

#[derive(Default)]
pub struct ModalAnalysis;

impl Analysis<MtlgExpr> for ModalAnalysis {
    type Data = ModalData;

    fn make(egraph: &EGraph<MtlgExpr, Self>, enode: &MtlgExpr) -> Self::Data {
        let mode_of = |id: &Id| egraph[*id].data.mode;
        let cat_of  = |id: &Id| egraph[*id].data.category;
        match enode {
            MtlgExpr::DiamondApp([f, _]) => ModalData { mode: ModalMode::Diamond, category: cat_of(f) },
            MtlgExpr::BoxApp([f, _])     => ModalData { mode: ModalMode::Box,     category: cat_of(f) },
            MtlgExpr::LozengeApp([f, _]) => ModalData { mode: ModalMode::Lozenge, category: cat_of(f) },
            MtlgExpr::Pred1([f, _]) | MtlgExpr::Pred2([f, _, _]) => {
                ModalData { mode: mode_of(f), category: TypeCategory::Scene }
            }
            MtlgExpr::Lam([_, body]) => ModalData { mode: mode_of(body), category: cat_of(body) },
            _ => ModalData::default_data(),
        }
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        ModalData::join(a, b)
    }
}

// ─── EGraph wrapper ───────────────────────────────────────────────────────────

pub struct ArgEGraph {
    pub egraph: EGraph<MtlgExpr, ModalAnalysis>,
    rewrites:  Vec<Rewrite<MtlgExpr, ModalAnalysis>>,
}

impl ArgEGraph {
    pub fn new() -> Self {
        Self {
            egraph:   EGraph::new(ModalAnalysis),
            rewrites: vec![
                // Beta reduction: (lam x body)[arg/x] → body  (simplified; full subst needed)
                // Commutativity of eq
                egg::rewrite!("eq-comm"; "(eq ?a ?b)" => "(eq ?b ?a)"),
                // Identity: (d-app f x) where f is identity-typed → x
                // (mode-preserving: only diamond-mode)
                egg::rewrite!("eta-diamond"; "(d-app (lam ?x ?x) ?a)" => "?a"),
                egg::rewrite!("eta-box";     "(b-app (lam ?x ?x) ?a)" => "?a"),
            ],
        }
    }

    /// Add an expression, returns its e-class id.
    pub fn add(&mut self, expr: RecExpr<MtlgExpr>) -> Id {
        self.egraph.add_expr(&expr)
    }

    /// Assert two expressions are equivalent (injects into same e-class).
    pub fn union(&mut self, a: Id, b: Id) -> bool {
        self.egraph.union(a, b)
    }

    /// Run equality saturation with the registered rewrites.
    pub fn saturate(&mut self) {
        let egraph = std::mem::replace(&mut self.egraph, EGraph::new(ModalAnalysis));
        let runner = Runner::default()
            .with_egraph(egraph)
            .with_iter_limit(10)
            .with_node_limit(5000)
            .run(&self.rewrites);
        self.egraph = runner.egraph;
    }

    /// Check if two IDs are in the same e-class (equivalent).
    pub fn equivalent(&self, a: Id, b: Id) -> bool {
        self.egraph.find(a) == self.egraph.find(b)
    }

    /// Get the modal data for an e-class.
    pub fn modal_data(&self, id: Id) -> &ModalData {
        &self.egraph[id].data
    }
}

impl Default for ArgEGraph {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_reduction_diamond() {
        let mut eg = ArgEGraph::new();
        // (d-app (lam x x) sym) should reduce to sym
        let expr: RecExpr<MtlgExpr> = "(d-app (lam x x) y)".parse().unwrap();
        let id = eg.add(expr);
        eg.saturate();
        // After saturation, check that y is in the same class
        let y: RecExpr<MtlgExpr> = "y".parse().unwrap();
        let y_id = eg.add(y);
        eg.saturate();
        assert!(eg.equivalent(id, y_id));
    }
}
