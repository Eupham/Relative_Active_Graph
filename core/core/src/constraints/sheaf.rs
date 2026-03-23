//! Discrete sheaf global-section consistency check over G(s).
//!
//! For each directed triangle (u → v → w, and u → w) in G(s), the restriction
//! maps along both paths must produce the same expected type at w. A violation
//! means the two derivation paths are modally inconsistent at their merge point.
//!
//! This is a global-section existence check, NOT H¹ computation.
//! True H¹ requires coboundary operators over a cochain complex
//! (see Curry, Ghrist & Robinson 2012 for the full algebraic topology treatment).

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, ModalType, ModalMode, TypeCategory, Direction};
use crate::arg::{ArgGraph, ArgEdge};
use petgraph::visit::EdgeRef;

/// Restriction map along one directed edge.
fn restriction_map(edge: &ArgEdge, source_type: ModalType) -> ModalType {
    match edge.modal_mode {
        ModalMode::Diamond => source_type.apply().unwrap_or(source_type),
        ModalMode::Box     => source_type,
        ModalMode::Lozenge => ModalType {
            mode:      ModalMode::Lozenge,
            category:  source_type.category,
            arity:     source_type.arity,
            direction: match source_type.direction {
                Direction::Right => Direction::Left,
                Direction::Left  => Direction::Right,
            },
        },
    }
}

/// Two types are consistent if their mode and category agree.
/// Arity differences are acceptable: partial application produces a different
/// saturation level but the same modal character.
fn types_consistent(a: ModalType, b: ModalType) -> bool {
    a.mode == b.mode && a.category == b.category
}

#[derive(Debug)]
pub struct CoherenceViolation {
    pub node_u:         NodeId,
    pub node_v:         NodeId,
    pub node_w:         NodeId,
    pub via_uv_vw_type: ModalType,
    pub via_uw_type:    ModalType,
}

/// Result of the sheaf consistency check.
///
/// `violation_count` is the number of triangles where the two restriction-map
/// paths produce different modal types at their shared target.
/// Zero violations means global sections exist for the observed triangles.
#[derive(Debug)]
pub struct SheafResult {
    /// Number of triangle coherence violations.
    /// This is NOT a cohomology rank — it is a raw inconsistency count.
    pub violation_count: usize,
    pub violations:      Vec<CoherenceViolation>,
}

impl SheafResult {
    pub fn is_coherent(&self) -> bool { self.violation_count == 0 }
}

pub fn check_sheaf_coherence(
    graph:  &ArgGraph,
    stalks: &HashMap<NodeId, ModalType>,
) -> SheafResult {
    let mut violations = Vec::new();
    let node_indices: Vec<_> = graph.node_indices().collect();

    for &u_idx in &node_indices {
        let u_id   = graph[u_idx].id;
        let u_type = match stalks.get(&u_id) { Some(t) => *t, None => continue };

        for uv in graph.edges(u_idx) {
            let v_idx            = uv.target();
            let v_id             = graph[v_idx].id;
            let type_at_v_via_uv = restriction_map(uv.weight(), u_type);

            for vw in graph.edges(v_idx) {
                let w_idx = vw.target();
                let w_id  = graph[w_idx].id;

                if let Some(uw) = graph.edges(u_idx).find(|e| e.target() == w_idx) {
                    let via_path = restriction_map(vw.weight(), type_at_v_via_uv);
                    let direct   = restriction_map(uw.weight(), u_type);

                    if !types_consistent(via_path, direct) {
                        violations.push(CoherenceViolation {
                            node_u: u_id,
                            node_v: v_id,
                            node_w: w_id,
                            via_uv_vw_type: via_path,
                            via_uw_type:    direct,
                        });
                    }
                }
            }
        }
    }

    SheafResult { violation_count: violations.len(), violations }
}

/// Build stalk map from the ARG graph's node modal types.
pub fn build_stalks(graph: &ArgGraph) -> HashMap<NodeId, ModalType> {
    graph.node_indices()
        .map(|idx| { let n = &graph[idx]; (n.id, n.mtlg_type) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalMode, TypeCategory, ModalType, Direction};
    use crate::arg::{ArgNode, NodeClass, ArgEdge, EdgeClass};
    use crate::arg::search::ArgGraph;
    use petgraph::stable_graph::StableGraph;

    fn diamond_type() -> ModalType {
        ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right)
    }

    #[test]
    fn coherent_triangle_passes() {
        let mut g: ArgGraph = StableGraph::new();
        let u_type  = diamond_type();
        let applied = u_type.apply().unwrap();

        let mut n_u = ArgNode::new(1, NodeClass::DEFAULT, u_type, (0,0));
        let mut n_v = ArgNode::new(2, NodeClass::DEFAULT, applied, (0,0));
        let mut n_w = ArgNode::new(3, NodeClass::DEFAULT, applied.apply().unwrap_or(applied), (0,0));
        n_u.atms_label = 0b111; n_v.atms_label = 0b111; n_w.atms_label = 0b111;

        let u = g.add_node(n_u);
        let v = g.add_node(n_v);
        let w = g.add_node(n_w);
        g.add_edge(u, v, ArgEdge::new(1, 1, 2, EdgeClass::DEFAULT, ModalMode::Diamond));
        g.add_edge(v, w, ArgEdge::new(2, 2, 3, EdgeClass::DEFAULT, ModalMode::Diamond));
        g.add_edge(u, w, ArgEdge::new(3, 1, 3, EdgeClass::DEFAULT, ModalMode::Diamond));

        let stalks = build_stalks(&g);
        let result = check_sheaf_coherence(&g, &stalks);
        assert!(result.is_coherent(), "violations: {:?}", result.violations);
    }

    #[test]
    fn violation_count_not_h1() {
        // Explicit documentation test: verify the field name and semantics.
        let result = SheafResult { violation_count: 0, violations: vec![] };
        assert!(result.is_coherent());
        // The field is violation_count, not h1_norm.
        // Any code referencing h1_norm will fail to compile, which is the intent.
        let _ = result.violation_count;
    }
}
