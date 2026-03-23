//! TypeNormalizer: applies KBC-completed Ruler rules as type rewrites to an ARG.

use crate::types::{ModalMode, TypeCategory};
use crate::arg::ArgGraph;
use crate::rules::ruler_bridge::MtlgRule;

/// Lexicographic path ordering for modal types.
/// Diamond(0) < Box(1) < Lozenge(2), then category_id, then arity.
/// A rule lhs → rhs is valid only if lhs > rhs under this ordering.
pub fn lpo_gt(
    lhs: (ModalMode, TypeCategory, u8),
    rhs: (ModalMode, TypeCategory, u8),
) -> bool {
    let rank = |m: ModalMode| -> u32 {
        match m { ModalMode::Diamond => 0, ModalMode::Box => 1, ModalMode::Lozenge => 2 }
    };
    (rank(lhs.0), lhs.1.0, lhs.2 as u32) > (rank(rhs.0), rhs.1.0, rhs.2 as u32)
}

/// Normalize a type to its normal form under a set of oriented rules.
/// Terminates because rules are LPO-oriented (each step strictly decreases).
pub fn normalize_type(
    mode:  ModalMode,
    cat:   TypeCategory,
    rules: &[MtlgRule],
) -> (ModalMode, TypeCategory) {
    let mut cur_mode = mode;
    let mut cur_cat  = cat;
    loop {
        match rules.iter().find(|r| r.lhs_mode == cur_mode && r.lhs_cat == cur_cat) {
            Some(r) => { cur_mode = r.rhs_mode; cur_cat = r.rhs_cat; }
            None    => break,
        }
    }
    (cur_mode, cur_cat)
}

/// Applies Ruler rules as type rewrites to every node in an ARG.
pub struct TypeNormalizer {
    rules: Vec<MtlgRule>,
}

impl TypeNormalizer {
    pub fn new(rules: Vec<MtlgRule>) -> Self { Self { rules } }

    pub fn apply_to_graph(&self, graph: &mut ArgGraph) {
        let node_ids: Vec<u64> = graph.node_indices().map(|i| graph[i].id).collect();
        for nid in node_ids {
            if let Some(ni) = graph.node_indices().find(|&i| graph[i].id == nid) {
                let arity = graph[ni].mtlg_type.arity;
                let (new_mode, new_cat) = normalize_type(
                    graph[ni].mtlg_type.mode,
                    graph[ni].mtlg_type.category,
                    &self.rules,
                );
                graph[ni].mtlg_type.mode     = new_mode;
                graph[ni].mtlg_type.category = new_cat;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalMode, TypeCategory};
    use crate::rules::ruler_bridge::MtlgRule;

    fn rule(lm: ModalMode, lc: u32, rm: ModalMode, rc: u32) -> MtlgRule {
        MtlgRule { id: 0, name: "t".into(), lhs_mode: lm, lhs_cat: TypeCategory(lc),
                   rhs_mode: rm, rhs_cat: TypeCategory(rc), confidence: 0.9, support: 20 }
    }

    #[test]
    fn normalizes_single_rule() {
        let rules = vec![rule(ModalMode::Box, 3, ModalMode::Diamond, 1)];
        let (m, c) = normalize_type(ModalMode::Box, TypeCategory(3), &rules);
        assert_eq!(m, ModalMode::Diamond);
        assert_eq!(c, TypeCategory(1));
    }

    #[test]
    fn lpo_box_gt_diamond() {
        assert!(lpo_gt((ModalMode::Box, TypeCategory(1), 0), (ModalMode::Diamond, TypeCategory(1), 0)));
    }
}
