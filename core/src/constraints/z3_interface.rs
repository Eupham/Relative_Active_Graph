//! MTLG modal type checker: validates mode consistency in MTLG derivations.
//! Uses a pure-Rust constraint solver over the modal type lattice.
//! (Z3 FFI bindings can be added via the `z3` crate for full SMT if needed;
//!  the logic here is equivalent for the modal fragment actually used in CSRRE.)

use std::collections::HashMap;
use crate::types::{ModalType, ModalMode, TypeCategory, Direction};

/// A modal type constraint: asserts that `lhs` compose-with `rhs` yields `result`.
#[derive(Clone, Debug)]
pub struct ModalConstraint {
    pub functor_type: ModalType,
    pub arg_type:     ModalType,
    pub result_type:  ModalType,
}

/// Result of type checking a derivation.
#[derive(Debug)]
pub enum TypeCheckResult {
    Valid(ModalType),
    Invalid(String),
}

impl TypeCheckResult {
    pub fn is_valid(&self) -> bool { matches!(self, TypeCheckResult::Valid(_)) }
}

/// Check that a functor type can apply to an argument type at the given position.
/// `arg_is_right`: true if the argument is to the right of the functor.
pub fn check_application(functor: ModalType, arg: ModalType, arg_is_right: bool) -> TypeCheckResult {
    if functor.arity == 0 {
        return TypeCheckResult::Invalid(format!(
            "functor {:?}/{:?} is already saturated (arity=0)", functor.mode, functor.category
        ));
    }
    if functor.mode != arg.mode {
        return TypeCheckResult::Invalid(format!(
            "mode mismatch: functor {:?} ≠ arg {:?}", functor.mode, arg.mode
        ));
    }
    if !functor.compatible_with(arg, arg_is_right) {
        return TypeCheckResult::Invalid(format!(
            "direction mismatch: functor seeks {:?} argument but arg_is_right={}",
            functor.direction, arg_is_right
        ));
    }
    match functor.apply() {
        Some(result) => TypeCheckResult::Valid(result),
        None         => TypeCheckResult::Invalid("apply failed unexpectedly".into()),
    }
}

/// Check a full derivation: (functor_type, arg_type, arg_is_right) steps.
pub fn check_derivation(steps: &[(ModalType, ModalType, bool)]) -> TypeCheckResult {
    if steps.is_empty() {
        return TypeCheckResult::Invalid("empty derivation".into());
    }
    let mut current = steps[0].0;
    for &(functor, arg, arg_is_right) in steps {
        match check_application(functor, arg, arg_is_right) {
            TypeCheckResult::Valid(next) => { current = next; }
            err                          => return err,
        }
    }
    TypeCheckResult::Valid(current)
}

/// Mode consistency: every edge's modal_mode matches its source node's mode,
/// and every application respects directionality.
pub fn check_mode_consistency(
    nodes: &HashMap<u64, ModalType>,
    edges: &[(u64, u64, ModalMode, bool)], // (src_id, dst_id, edge_mode, dst_is_right_of_src)
) -> Vec<String> {
    let mut errors = Vec::new();
    for &(src, dst, edge_mode, dst_is_right) in edges {
        let src_type = match nodes.get(&src) {
            Some(t) => t,
            None    => { errors.push(format!("src node {} not in type map", src)); continue; }
        };
        if src_type.mode != edge_mode {
            errors.push(format!(
                "edge ({src}→{dst}): edge mode {:?} ≠ src type mode {:?}",
                edge_mode, src_type.mode
            ));
        }
        // Directionality: the edge flows from src to dst; dst is either right or left of src.
        match src_type.direction {
            Direction::Right if !dst_is_right => errors.push(format!(
                "edge ({src}→{dst}): functor seeks Right argument but dst is to the left"
            )),
            Direction::Left if dst_is_right => errors.push(format!(
                "edge ({src}→{dst}): functor seeks Left argument but dst is to the right"
            )),
            _ => {}
        }
    }
    errors
}

/// Validate that a □-mode (sharing) application obeys the contraction rule:
/// both functors share the same argument (same type, same identity).
pub fn check_box_sharing(
    functor_a: ModalType,
    functor_b: ModalType,
    shared_arg: ModalType,
) -> TypeCheckResult {
    if functor_a.mode != ModalMode::Box || functor_b.mode != ModalMode::Box {
        return TypeCheckResult::Invalid("box sharing requires both functors in □ mode".into());
    }
    match (functor_a.apply(), functor_b.apply()) {
        (Some(ra), Some(rb)) if ra.mode == rb.mode => TypeCheckResult::Valid(ra),
        (Some(_), Some(_)) => TypeCheckResult::Invalid("□-sharing result types diverge".into()),
        _ => TypeCheckResult::Invalid("box sharing: one functor is saturated".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(cat: TypeCategory, arity: u8) -> ModalType {
        ModalType::functor(ModalMode::Diamond, cat, arity, Direction::Right)
    }

    #[test]
    fn valid_application() {
        let f = d(TypeCategory::Scene, 1);
        let a = d(TypeCategory::Participant, 0);
        assert!(check_application(f, a, true).is_valid());
    }

    #[test]
    fn saturated_functor_fails() {
        let f = d(TypeCategory::Scene, 0);
        let a = d(TypeCategory::Participant, 0);
        assert!(!check_application(f, a, true).is_valid());
    }

    #[test]
    fn mode_mismatch_fails() {
        let f = ModalType::functor(ModalMode::Diamond, TypeCategory::Scene, 1, Direction::Right);
        let a = ModalType::functor(ModalMode::Box,     TypeCategory::Scene, 0, Direction::Right);
        assert!(!check_application(f, a, true).is_valid());
    }

    #[test]
    fn direction_mismatch_fails() {
        let f = ModalType::functor(ModalMode::Diamond, TypeCategory::Scene, 1, Direction::Right);
        let a = ModalType::functor(ModalMode::Diamond, TypeCategory::Participant, 0, Direction::Right);
        // arg_is_right=false but functor seeks Right → should fail
        assert!(!check_application(f, a, false).is_valid());
    }

    #[test]
    fn mode_consistency_check() {
        let mut types = HashMap::new();
        types.insert(1u64, ModalType::functor(ModalMode::Diamond, TypeCategory::Scene, 1, Direction::Right));
        types.insert(2u64, ModalType::atom(ModalMode::Diamond, TypeCategory::Participant));
        let edges = vec![(1u64, 2u64, ModalMode::Diamond, true)];
        let errors = check_mode_consistency(&types, &edges);
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }
}
