//! MTLG Semantics module — merged from (AMR + DRT + UCCA) as unified derivation.
//! sentence_level() → λ-term / AMR-like proposition graph (one TR scope).
//! discourse_level() → DRS update (inter-TR referent accumulation).
//! modal_type_extract() → UCCA category label from MTLG derivation.

use std::collections::HashMap;
use crate::types::{NodeId, ModalType, ModalMode, TypeCategory};
use crate::arg::transient_repr::RepContent;

// ─── Lambda terms ──────────────────────────────────────────────────────────────

/// Simply-typed lambda calculus extended with predicates (HOL extensional fragment).
/// Intensional constructions (belief, modals) require explicit wrapping via Intension.
#[derive(Clone, Debug, PartialEq)]
pub enum LambdaTerm {
    Var(String),
    Const(String),
    App(Box<LambdaTerm>, Box<LambdaTerm>),
    Abs(String, ModalType, Box<LambdaTerm>),
    Pred(String, Vec<LambdaTerm>),
    /// Intensional wrapper: holds a proposition as a function over possible worlds.
    Intension(Box<LambdaTerm>),
}

impl LambdaTerm {
    /// Beta-reduce one step: (λx.body) arg → body[arg/x].
    pub fn beta_reduce(self) -> Self {
        match self {
            LambdaTerm::App(f, arg) => {
                if let LambdaTerm::Abs(var, _, body) = *f {
                    substitute(*body, &var, &arg)
                } else {
                    LambdaTerm::App(f, arg)
                }
            }
            other => other,
        }
    }

    /// Fully beta-normalize (up to depth 20 to avoid infinite loops).
    pub fn normalize(self) -> Self {
        let mut term = self;
        for _ in 0..20 {
            let reduced = term.clone().beta_reduce();
            if reduced == term { break; }
            term = reduced;
        }
        term
    }

    /// Serialize to a human-readable string.
    pub fn display(&self) -> String {
        match self {
            LambdaTerm::Var(v)      => v.clone(),
            LambdaTerm::Const(c)    => c.clone(),
            LambdaTerm::App(f, a)   => format!("({} {})", f.display(), a.display()),
            LambdaTerm::Abs(v, _, b) => format!("(λ{}.{})", v, b.display()),
            LambdaTerm::Pred(p, args) => format!("{}({})", p, args.iter().map(|a| a.display()).collect::<Vec<_>>().join(",")),
            LambdaTerm::Intension(t) => format!("^{}", t.display()),
        }
    }

    /// Extract UCCA category from the outermost functor's modal type.
    pub fn ucca_category(&self, type_map: &HashMap<String, ModalType>) -> TypeCategory {
        match self {
            LambdaTerm::Pred(name, _) => {
                type_map.get(name).map_or(TypeCategory::Scene, |mt| mt.category)
            }
            LambdaTerm::App(f, _) => f.ucca_category(type_map),
            LambdaTerm::Abs(_, ty, _) => ty.category,
            _ => TypeCategory::Scene,
        }
    }
}

fn substitute(term: LambdaTerm, var: &str, replacement: &LambdaTerm) -> LambdaTerm {
    match term {
        LambdaTerm::Var(v) if v == var => replacement.clone(),
        LambdaTerm::Var(v) => LambdaTerm::Var(v),
        LambdaTerm::Const(c) => LambdaTerm::Const(c),
        LambdaTerm::App(f, a) => LambdaTerm::App(
            Box::new(substitute(*f, var, replacement)),
            Box::new(substitute(*a, var, replacement)),
        ),
        LambdaTerm::Abs(v, ty, body) if v != var => {
            LambdaTerm::Abs(v, ty, Box::new(substitute(*body, var, replacement)))
        }
        other => other,
    }
}

// ─── AMR-style proposition (sentence level) ───────────────────────────────────

/// Sentence-level output: predicate-argument graph (AMR-like).
#[derive(Clone, Debug)]
pub struct PropositionGraph {
    pub root:     String,
    pub roles:    Vec<(String, String)>, // (role_label, argument_node)
    pub lambda_str: String,
}

impl PropositionGraph {
    pub fn from_lambda(term: &LambdaTerm) -> Self {
        let root = match term {
            LambdaTerm::Pred(p, _) => p.clone(),
            LambdaTerm::App(f, _) => {
                if let LambdaTerm::Pred(p, _) = f.as_ref() { p.clone() } else { "ROOT".into() }
            }
            _ => "ROOT".into(),
        };
        let roles = extract_roles(term);
        Self { root, roles, lambda_str: term.display() }
    }
}

fn extract_roles(term: &LambdaTerm) -> Vec<(String, String)> {
    match term {
        LambdaTerm::Pred(_, args) => args.iter().enumerate()
            .map(|(i, a)| (format!("ARG{}", i), a.display()))
            .collect(),
        LambdaTerm::App(f, a) => {
            let mut roles = extract_roles(f);
            roles.push(("ARG".into(), a.display()));
            roles
        }
        _ => vec![],
    }
}

// ─── DRS update (discourse level) ────────────────────────────────────────────

/// Discourse-level output: DRS update (new referents + conditions).
/// Existential variables from the λ-term → new DRS referents.
#[derive(Clone, Debug, Default)]
pub struct DrsUpdate {
    pub new_referents:  Vec<String>,
    pub new_conditions: Vec<String>,
}

impl DrsUpdate {
    pub fn from_lambda(term: &LambdaTerm, parent_referents: &[String]) -> Self {
        let mut update = DrsUpdate::default();
        collect_referents(term, parent_referents, &mut update);
        update
    }

    pub fn to_rep_content(&self) -> RepContent {
        RepContent::DrsUpdate {
            referents:  self.new_referents.clone(),
            conditions: self.new_conditions.clone(),
        }
    }
}

fn collect_referents(term: &LambdaTerm, existing: &[String], update: &mut DrsUpdate) {
    match term {
        LambdaTerm::Abs(var, _, body) => {
            if !existing.contains(var) && !update.new_referents.contains(var) {
                update.new_referents.push(var.clone());
            }
            collect_referents(body, existing, update);
        }
        LambdaTerm::Pred(name, args) => {
            let arg_strs: Vec<String> = args.iter().map(|a| a.display()).collect();
            update.new_conditions.push(format!("{}({})", name, arg_strs.join(",")));
            for a in args { collect_referents(a, existing, update); }
        }
        LambdaTerm::App(f, a) => {
            collect_referents(f, existing, update);
            collect_referents(a, existing, update);
        }
        LambdaTerm::Intension(t) => collect_referents(t, existing, update),
        _ => {}
    }
}

// ─── MTLG Semantics module ────────────────────────────────────────────────────

pub struct MtlgSemantics {
    /// Per-predicate modal types (from LCS induction).
    pub type_map: HashMap<String, ModalType>,
}

impl MtlgSemantics {
    pub fn new() -> Self {
        Self { type_map: HashMap::new() }
    }

    pub fn register_type(&mut self, predicate: String, modal_type: ModalType) {
        self.type_map.insert(predicate, modal_type);
    }

    /// sentence_level: evaluate the λ-term and produce an AMR-like proposition graph.
    pub fn sentence_level(&self, term: LambdaTerm) -> PropositionGraph {
        let normalized = term.normalize();
        PropositionGraph::from_lambda(&normalized)
    }

    /// discourse_level: extract new DRS referents from the λ-term given parent context.
    pub fn discourse_level(&self, term: &LambdaTerm, parent_refs: &[String]) -> DrsUpdate {
        DrsUpdate::from_lambda(term, parent_refs)
    }

    /// modal_type_extract: derive the UCCA category label from the derivation's outermost type.
    pub fn modal_type_extract(&self, term: &LambdaTerm) -> TypeCategory {
        term.ucca_category(&self.type_map)
    }

    /// Compose two MTLG types: f_type ◇ arg_type → result_type (if compatible).
    pub fn compose(&self, f_type: ModalType, arg_type: ModalType) -> Option<ModalType> {
        if f_type.compatible_with(arg_type) { f_type.apply() } else { None }
    }
}

impl Default for MtlgSemantics {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModalMode;

    #[test]
    fn beta_reduce_basic() {
        let mt = ModalType::atom(ModalMode::Diamond, TypeCategory::Scene);
        let term = LambdaTerm::App(
            Box::new(LambdaTerm::Abs("x".into(), mt, Box::new(LambdaTerm::Var("x".into())))),
            Box::new(LambdaTerm::Const("alice".into())),
        );
        let result = term.beta_reduce();
        assert_eq!(result, LambdaTerm::Const("alice".into()));
    }

    #[test]
    fn drs_update_extracts_vars() {
        let mt = ModalType::atom(ModalMode::Diamond, TypeCategory::Participant);
        let term = LambdaTerm::Abs("x".into(), mt, Box::new(LambdaTerm::Pred("run".into(), vec![LambdaTerm::Var("x".into())])));
        let update = DrsUpdate::from_lambda(&term, &[]);
        assert!(update.new_referents.contains(&"x".into()));
        assert!(!update.new_conditions.is_empty());
    }

    #[test]
    fn sentence_level_proposition() {
        let sem = MtlgSemantics::new();
        let term = LambdaTerm::Pred("run".into(), vec![LambdaTerm::Const("alice".into())]);
        let prop = sem.sentence_level(term);
        assert_eq!(prop.root, "run");
        assert!(!prop.roles.is_empty());
    }
}
