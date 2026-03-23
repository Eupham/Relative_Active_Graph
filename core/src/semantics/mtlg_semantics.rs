//! MTLG Semantics: sentence_level (AMR-like), discourse_level (DRS), modal type extraction.
//! Variable capture in substitute() is correctly handled via free variable analysis.

use std::collections::HashMap;
use crate::types::{NodeId, ModalType, ModalMode, TypeCategory};
use crate::arg::transient_repr::RepContent;

// ─── Lambda terms ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum LambdaTerm {
    Var(String),
    Const(String),
    App(Box<LambdaTerm>, Box<LambdaTerm>),
    Abs(String, ModalType, Box<LambdaTerm>),
    Pred(String, Vec<LambdaTerm>),
    Intension(Box<LambdaTerm>),
}

impl LambdaTerm {
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

    pub fn normalize(self) -> Self {
        let mut term = self;
        for _ in 0..200 {
            let reduced = term.clone().beta_reduce();
            if reduced == term { break; }
            term = reduced;
        }
        term
    }

    pub fn display(&self) -> String {
        match self {
            LambdaTerm::Var(v)         => v.clone(),
            LambdaTerm::Const(c)       => c.clone(),
            LambdaTerm::App(f, a)      => format!("({} {})", f.display(), a.display()),
            LambdaTerm::Abs(v, _, b)   => format!("(λ{}.{})", v, b.display()),
            LambdaTerm::Pred(p, args)  =>
                format!("{}({})", p, args.iter().map(|a| a.display()).collect::<Vec<_>>().join(",")),
            LambdaTerm::Intension(t)   => format!("^{}", t.display()),
        }
    }

    pub fn ucca_category(&self, type_map: &HashMap<String, ModalType>) -> TypeCategory {
        match self {
            LambdaTerm::Pred(name, _)  => type_map.get(name).map_or(TypeCategory::DEFAULT, |mt| mt.category),
            LambdaTerm::App(f, _)      => f.ucca_category(type_map),
            LambdaTerm::Abs(_, ty, _)  => ty.category,
            _ => TypeCategory::DEFAULT,
        }
    }
}

fn free_vars(term: &LambdaTerm) -> std::collections::HashSet<String> {
    match term {
        LambdaTerm::Var(v)        => std::iter::once(v.clone()).collect(),
        LambdaTerm::Const(_)      => Default::default(),
        LambdaTerm::App(f, a)     => { let mut s = free_vars(f); s.extend(free_vars(a)); s }
        LambdaTerm::Abs(v, _, b)  => { let mut s = free_vars(b); s.remove(v); s }
        LambdaTerm::Pred(_, args) => args.iter().flat_map(free_vars).collect(),
        LambdaTerm::Intension(t)  => free_vars(t),
    }
}

fn fresh_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn substitute(term: LambdaTerm, var: &str, replacement: &LambdaTerm) -> LambdaTerm {
    match term {
        LambdaTerm::Var(v) if v == var => replacement.clone(),
        LambdaTerm::Var(v)             => LambdaTerm::Var(v),
        LambdaTerm::Const(c)           => LambdaTerm::Const(c),
        LambdaTerm::App(f, a)          => LambdaTerm::App(
            Box::new(substitute(*f, var, replacement)),
            Box::new(substitute(*a, var, replacement)),
        ),
        LambdaTerm::Abs(v, ty, body) => {
            if v == var {
                LambdaTerm::Abs(v, ty, body)
            } else if free_vars(replacement).contains(&v) {
                let fresh = format!("{}_r{}", v, fresh_id());
                let renamed = substitute(*body, &v, &LambdaTerm::Var(fresh.clone()));
                LambdaTerm::Abs(fresh, ty, Box::new(substitute(renamed, var, replacement)))
            } else {
                LambdaTerm::Abs(v, ty, Box::new(substitute(*body, var, replacement)))
            }
        }
        LambdaTerm::Pred(p, args) => LambdaTerm::Pred(
            p, args.into_iter().map(|a| substitute(a, var, replacement)).collect(),
        ),
        LambdaTerm::Intension(t) => LambdaTerm::Intension(Box::new(substitute(*t, var, replacement))),
    }
}

// ─── Proposition ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PropositionGraph {
    pub root:       String,
    pub roles:      Vec<(String, String)>,
    pub lambda_str: String,
}

impl PropositionGraph {
    pub fn from_lambda(term: &LambdaTerm) -> Self {
        let root = match term {
            LambdaTerm::Pred(p, _) => p.clone(),
            LambdaTerm::App(f, _)  => {
                if let LambdaTerm::Pred(p, _) = f.as_ref() { p.clone() } else { "ROOT".into() }
            }
            _ => "ROOT".into(),
        };
        Self { root, roles: extract_roles(term), lambda_str: term.display() }
    }
}

fn extract_roles(term: &LambdaTerm) -> Vec<(String, String)> {
    match term {
        LambdaTerm::Pred(_, args) => args.iter().enumerate()
            .map(|(i, a)| (format!("ARG{}", i), a.display())).collect(),
        LambdaTerm::App(f, a) => {
            let mut roles = extract_roles(f);
            roles.push(("ARG".into(), a.display()));
            roles
        }
        _ => vec![],
    }
}

// ─── DRS ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct DrsUpdate {
    pub new_referents:  Vec<String>,
    pub new_conditions: Vec<String>,
}

impl DrsUpdate {
    pub fn from_lambda(term: &LambdaTerm, parent_refs: &[String]) -> Self {
        let mut referents  = Vec::new();
        let mut conditions = Vec::new();
        collect_drs(term, &mut referents, &mut conditions);
        referents.retain(|r| !parent_refs.contains(r));
        Self { new_referents: referents, new_conditions: conditions }
    }
}

fn collect_drs(term: &LambdaTerm, refs: &mut Vec<String>, conds: &mut Vec<String>) {
    match term {
        LambdaTerm::Abs(v, _, body) => {
            refs.push(v.clone());
            collect_drs(body, refs, conds);
        }
        LambdaTerm::Pred(p, args) => {
            let arg_strs: Vec<String> = args.iter().map(|a| a.display()).collect();
            conds.push(format!("{}({})", p, arg_strs.join(",")));
        }
        LambdaTerm::App(f, a) => {
            collect_drs(f, refs, conds);
            collect_drs(a, refs, conds);
        }
        _ => {}
    }
}

// ─── MtlgSemantics ───────────────────────────────────────────────────────────

pub struct MtlgSemantics {
    pub type_map: HashMap<String, ModalType>,
}

impl MtlgSemantics {
    pub fn new() -> Self { Self { type_map: HashMap::new() } }

    pub fn register_type(&mut self, predicate: String, ty: ModalType) {
        self.type_map.insert(predicate, ty);
    }

    pub fn sentence_level(&self, term: LambdaTerm) -> PropositionGraph {
        PropositionGraph::from_lambda(&term.normalize())
    }

    pub fn discourse_level(&self, term: &LambdaTerm, parent_refs: &[String]) -> DrsUpdate {
        DrsUpdate::from_lambda(term, parent_refs)
    }

    pub fn modal_type_extract(&self, term: &LambdaTerm) -> TypeCategory {
        term.ucca_category(&self.type_map)
    }

    pub fn compose(&self, f_type: ModalType, arg_type: ModalType) -> Option<ModalType> {
        if f_type.compatible_with(arg_type, true) { f_type.apply() } else { None }
    }
}

impl Default for MtlgSemantics { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModalMode;

    #[test]
    fn beta_reduce_basic() {
        let mt   = ModalType::atom(ModalMode::Diamond, TypeCategory::DEFAULT);
        let term = LambdaTerm::App(
            Box::new(LambdaTerm::Abs("x".into(), mt, Box::new(LambdaTerm::Var("x".into())))),
            Box::new(LambdaTerm::Const("alice".into())),
        );
        assert_eq!(term.beta_reduce(), LambdaTerm::Const("alice".into()));
    }

    #[test]
    fn substitute_no_capture() {
        let mt   = ModalType::atom(ModalMode::Diamond, TypeCategory::DEFAULT);
        // (λy. y) applied to x: should give x, not capture
        let term = LambdaTerm::Abs("y".into(), mt, Box::new(LambdaTerm::Var("y".into())));
        let app  = LambdaTerm::App(Box::new(term), Box::new(LambdaTerm::Var("x".into())));
        assert_eq!(app.beta_reduce(), LambdaTerm::Var("x".into()));
    }

    #[test]
    fn sentence_level_proposition() {
        let sem  = MtlgSemantics::new();
        let term = LambdaTerm::Pred("run".into(), vec![LambdaTerm::Const("alice".into())]);
        let prop = sem.sentence_level(term);
        assert_eq!(prop.root, "run");
        assert!(!prop.roles.is_empty());
    }
}
