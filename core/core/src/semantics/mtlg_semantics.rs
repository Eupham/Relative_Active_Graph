//! MTLG Semantics: sentence_level (lambda/proposition graph), discourse_level
//! (DRS-structured referent sets), modal type extraction.
//!
//! Category labels (Scene, Process, State, Connector, Ground, Participant, Adverbial)
//! follow UCCA naming conventions (Abend & Rappoport 2013) but are assigned here by
//! bisimulation partition refinement over surface features — not by a trained UCCA
//! parser or annotation. The AMR-style role labels (ARG0, ARG1, …) used in
//! PropositionGraph are borrowed from PropBank/AMR conventions for readability;
//! the underlying representation is a plain lambda term, not an AMR graph.
//!
//! Full UCCA and AMR compliance are development targets, not current capabilities.
//!
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

/// Error returned when lambda normalization cannot reach normal form.
#[derive(Debug)]
pub enum ReductionError {
    NonTerminating(LambdaTerm),
}

impl std::fmt::Display for ReductionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lambda term does not reach normal form (possible cycle detected)")
    }
}

impl LambdaTerm {
    /// Single-step beta reduction (outermost leftmost redex, no strategy).
    #[deprecated(note = "use reduce_one_normal_order or normalize")]
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

    /// Reduce the leftmost-outermost β-redex (normal-order reduction). Returns None if no redex.
    pub fn reduce_one_normal_order(self) -> Option<LambdaTerm> {
        match self {
            LambdaTerm::App(f, arg) => {
                if let LambdaTerm::Abs(var, ty, body) = *f {
                    // Outermost redex — reduce it.
                    Some(substitute(*body, &var, &arg))
                } else {
                    let f_inner = *f;
                    if let Some(f_r) = f_inner.clone().reduce_one_normal_order() {
                        Some(LambdaTerm::App(Box::new(f_r), arg))
                    } else {
                        arg.reduce_one_normal_order()
                           .map(|a| LambdaTerm::App(Box::new(f_inner), Box::new(a)))
                    }
                }
            }
            LambdaTerm::Abs(var, ty, body) => {
                body.reduce_one_normal_order()
                    .map(|b| LambdaTerm::Abs(var, ty, Box::new(b)))
            }
            LambdaTerm::Pred(p, args) => {
                for (i, arg) in args.clone().into_iter().enumerate() {
                    if let Some(r) = arg.reduce_one_normal_order() {
                        let mut new_args = args;
                        new_args[i] = r;
                        return Some(LambdaTerm::Pred(p, new_args));
                    }
                }
                None
            }
            LambdaTerm::Intension(t) => {
                t.reduce_one_normal_order().map(|r| LambdaTerm::Intension(Box::new(r)))
            }
            LambdaTerm::Var(_) | LambdaTerm::Const(_) => None,
        }
    }

    /// Structural hash for cycle detection using FNV-1a.
    /// Delegates to token_types::fnv1a_64_bytes for consistent hashing.
    pub fn structural_hash(&self) -> u64 {
        let canonical = format!("{:?}", self);
        crate::lcs::fnv1a_64_bytes(canonical.as_bytes())
    }

    /// Normalize via normal-order reduction with structural hash cycle detection (§8).
    pub fn normalize(self) -> Result<LambdaTerm, ReductionError> {
        let mut term = self;
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        loop {
            let h = term.structural_hash();
            if !seen.insert(h) {
                return Err(ReductionError::NonTerminating(term));
            }
            let next = term.clone().reduce_one_normal_order();
            match next {
                None    => return Ok(term),
                Some(t) => { term = t; }
            }
        }
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
        let normalized = term.normalize().unwrap_or_else(|e| {
            log::warn!("normalize: {}", e);
            LambdaTerm::Const("_UNREDUCED".into())
        });
        PropositionGraph::from_lambda(&normalized)
    }

    pub fn discourse_level(&self, term: &LambdaTerm, parent_refs: &[String]) -> DrsUpdate {
        DrsUpdate::from_lambda(term, parent_refs)
    }

    pub fn modal_type_extract(&self, term: &LambdaTerm) -> TypeCategory {
        term.ucca_category(&self.type_map)
    }

    /// Equality-based compose (legacy). Use compose_with_meta for proper rule-checked composition.
    pub fn compose(&self, f_type: ModalType, arg_type: ModalType) -> Option<ModalType> {
        if f_type.compatible_with(arg_type, true) { f_type.apply() } else { None }
    }

    /// Rule-checked compose using MetaGrammarEngine (§12c).
    /// When query_composition returns None (no rule licenses the composition),
    /// the composition is rejected.
    pub fn compose_with_meta(
        &self,
        f_type:   ModalType,
        arg_type: ModalType,
        meta:     &crate::semantics::meta_grammar::MetaGrammarEngine,
    ) -> Option<ModalType> {
        use crate::semantics::meta_grammar::TypedFact;
        let fact_f   = TypedFact { category: f_type.category,   mode: f_type.mode,   direction: f_type.direction };
        let fact_arg = TypedFact { category: arg_type.category, mode: arg_type.mode, direction: arg_type.direction };
        meta.query_composition(fact_f, fact_arg).map(|result| {
            ModalType {
                category:  result.category,
                mode:      result.mode,
                direction: result.direction,
                arity:     f_type.arity.saturating_sub(1),
            }
        })
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
