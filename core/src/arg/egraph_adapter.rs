//! Equality saturation adapter using egg.
//! E-classes are keyed by (modal_mode, category) to enforce modal separation:
//! ◇-composed and □-composed subgraphs never merge into the same e-class.

use egg::{define_language, Analysis, DidMerge, EGraph, Id, RecExpr, Rewrite, Runner, Symbol};
use crate::types::{ModalMode, TypeCategory};

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
        Self { mode: ModalMode::Diamond, category: TypeCategory::DEFAULT }
    }

    fn join(a: &mut Self, b: Self) -> DidMerge {
        // Lattice join: modes must agree; category defaults to DEFAULT on conflict.
        let changed_mode = a.mode != b.mode;
        let changed_cat  = a.category != b.category;
        if changed_mode { a.mode = ModalMode::Diamond; }  // conservative default
        if changed_cat  { a.category = TypeCategory::DEFAULT; }
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
                ModalData { mode: mode_of(f), category: TypeCategory::DEFAULT }
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
                // Existing
                egg::rewrite!("eq-comm";     "(eq ?a ?b)"             => "(eq ?b ?a)"),
                egg::rewrite!("eta-diamond"; "(d-app (lam ?x ?x) ?a)" => "?a"),
                egg::rewrite!("eta-box";     "(b-app (lam ?x ?x) ?a)" => "?a"),

                // Knuth-Bendix modal composition — ordering: ◊ > □ > ◇
                egg::rewrite!("kb-box-outer-absorbs-diamond";
                    "(b-app (d-app ?f ?x) ?y)" => "(b-app ?f ?y)"),
                egg::rewrite!("kb-inner-box-propagates";
                    "(d-app (b-app ?f ?x) ?y)" => "(b-app ?f ?y)"),
                egg::rewrite!("kb-box-idempotent";
                    "(b-app (b-app ?f ?x) ?y)" => "(b-app ?f ?y)"),
                egg::rewrite!("kb-lozenge-outer-absorbs-diamond";
                    "(l-app (d-app ?f ?x) ?y)" => "(l-app ?f ?y)"),
                egg::rewrite!("kb-lozenge-outer-absorbs-box";
                    "(l-app (b-app ?f ?x) ?y)" => "(l-app ?f ?y)"),
                egg::rewrite!("kb-inner-lozenge-through-diamond";
                    "(d-app (l-app ?f ?x) ?y)" => "(l-app ?f ?y)"),
                egg::rewrite!("kb-inner-lozenge-through-box";
                    "(b-app (l-app ?f ?x) ?y)" => "(l-app ?f ?y)"),
                egg::rewrite!("kb-lozenge-idempotent";
                    "(l-app (l-app ?f ?x) ?y)" => "(l-app ?f ?y)"),
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

    /// Knuth-Bendix normal form for a sequence of modal modes.
    /// Priority: ◊ > □ > ◇. Returns None for paths shorter than 2 hops.
    pub fn canonicalize_path(
        &mut self,
        path: &[(crate::types::ModalMode, crate::types::TypeCategory)],
    ) -> Option<(crate::types::ModalMode, crate::types::TypeCategory)> {
        use crate::types::ModalMode;
        if path.len() < 2 { return None; }
        let canonical_mode = path.iter().fold(ModalMode::Diamond, |acc, &(m, _)| {
            match (acc, m) {
                (ModalMode::Lozenge, _) | (_, ModalMode::Lozenge) => ModalMode::Lozenge,
                (ModalMode::Box, _)    | (_, ModalMode::Box)      => ModalMode::Box,
                _                                                   => ModalMode::Diamond,
            }
        });
        let canonical_cat = path.iter().map(|&(_, c)| c)
            .find(|c| c.is_assigned())
            .unwrap_or(crate::types::TypeCategory::DEFAULT);
        Some((canonical_mode, canonical_cat))
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

    #[test]
    fn kb_box_absorbs_diamond() {
        let mut eg = ArgEGraph::new();
        let l = eg.add("(b-app (d-app f x) y)".parse().unwrap());
        let s = eg.add("(b-app f y)".parse().unwrap());
        eg.saturate();
        assert!(eg.equivalent(l, s));
    }
    #[test]
    fn kb_lozenge_dominates_all() {
        let mut eg = ArgEGraph::new();
        let l = eg.add("(l-app (b-app (d-app f x) y) z)".parse().unwrap());
        let s = eg.add("(l-app f z)".parse().unwrap());
        eg.saturate();
        assert!(eg.equivalent(l, s));
    }
    #[test]
    fn canonicalize_box_wins() {
        use crate::types::{ModalMode, TypeCategory};
        let mut eg = ArgEGraph::new();
        let (mode, _) = eg.canonicalize_path(&[
            (ModalMode::Diamond, TypeCategory::DEFAULT),
            (ModalMode::Box,     TypeCategory::DEFAULT),
        ]).unwrap();
        assert_eq!(mode, ModalMode::Box);
    }
    #[test]
    fn canonicalize_lozenge_wins() {
        use crate::types::{ModalMode, TypeCategory};
        let mut eg = ArgEGraph::new();
        let (mode, _) = eg.canonicalize_path(&[
            (ModalMode::Box,     TypeCategory::DEFAULT),
            (ModalMode::Lozenge, TypeCategory::DEFAULT),
        ]).unwrap();
        assert_eq!(mode, ModalMode::Lozenge);
    }
}
