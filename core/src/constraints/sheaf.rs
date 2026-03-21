//! Discrete sheaf Laplacian: checks modal-type stalk coherence at merge boundaries.
//! H¹ ≠ 0 means two derivation paths produce incompatible modal types at a shared node.
//!
//! For each triangle (u → v → w) in G(s): check that restriction maps compose consistently.
//! Stalk F(v) = ModalType at v. Restriction map r_{u→v}: F(u) → F(v).

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, ModalType, ModalMode, TypeCategory};
use crate::arg::{ArgGraph, ArgEdge, edge::EdgeType};
use petgraph::visit::EdgeRef;

/// Restriction map: given the modal type at the source, what is the expected type at target?
/// For ◇-mode edges: applies one argument (reduces arity by 1, preserves mode/category).
/// For □-mode edges: preserves arity (contraction — same resource used twice).
/// For ◊-mode edges: shifts to discontinuous mode.
fn restriction_map(edge: &ArgEdge, source_type: ModalType) -> ModalType {
    match edge.modal_mode {
        ModalMode::Diamond => source_type.apply().unwrap_or(source_type),
        ModalMode::Box     => source_type, // contraction: type is shared, not consumed
        ModalMode::Lozenge => ModalType {
            mode: ModalMode::Lozenge,
            category: source_type.category,
            arity: source_type.arity,
            rightward: !source_type.rightward, // discontinuous: reverses directionality
        },
    }
}

/// Coherence violation: two paths to the same node produce different expected types.
#[derive(Debug)]
pub struct CoherenceViolation {
    pub node_u:         NodeId,
    pub node_v:         NodeId,
    pub node_w:         NodeId,
    pub via_uv_vw_type: ModalType,
    pub via_uw_type:    ModalType,
}

/// Result of the sheaf coherence check.
#[derive(Debug)]
pub struct SheafResult {
    pub h1_norm:    f32,  // 0.0 = coherent; > 0.0 means |H¹| violations
    pub violations: Vec<CoherenceViolation>,
}

impl SheafResult {
    pub fn is_coherent(&self) -> bool { self.h1_norm == 0.0 }
}

/// Check sheaf coherence over G(s): find all triangles and verify restriction map consistency.
pub fn check_sheaf_coherence(
    graph:  &ArgGraph,
    stalks: &HashMap<NodeId, ModalType>,
) -> SheafResult {
    let mut violations = Vec::new();

    // Build adjacency: NodeId → [(neighbor NodeId, edge weight)]
    let node_indices: Vec<_> = graph.node_indices().collect();

    for &u_idx in &node_indices {
        let u_id = graph[u_idx].id;
        let u_type = match stalks.get(&u_id) { Some(t) => *t, None => continue };

        // Edges u → v
        for uv in graph.edges(u_idx) {
            let v_idx = uv.target();
            let v_id  = graph[v_idx].id;
            let type_at_v_via_uv = restriction_map(uv.weight(), u_type);

            // Edges v → w
            for vw in graph.edges(v_idx) {
                let w_idx = vw.target();
                let w_id  = graph[w_idx].id;

                // Look for direct edge u → w
                let uw_edge = graph.edges(u_idx).find(|e| e.target() == w_idx);
                if let Some(uw) = uw_edge {
                    // Two paths: u→v→w and u→w
                    let via_path = restriction_map(vw.weight(), type_at_v_via_uv);
                    let direct   = restriction_map(uw.weight(), u_type);

                    if !types_cohere(via_path, direct) {
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

    let h1_norm = violations.len() as f32;
    SheafResult { h1_norm, violations }
}

/// Two types cohere if they agree on mode and category (arity differences are acceptable
/// since arity represents remaining unsaturated arguments, which can differ by path).
fn types_cohere(a: ModalType, b: ModalType) -> bool {
    a.mode == b.mode && a.category == b.category
}

/// Build stalk map from the ARG graph's node modal types.
pub fn build_stalks(graph: &ArgGraph) -> HashMap<NodeId, ModalType> {
    graph.node_indices()
        .map(|idx| {
            let n = &graph[idx];
            (n.id, n.mtlg_type)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalMode, TypeCategory, ModalType};
    use crate::arg::{ArgNode, NodeType, ArgEdge, EdgeType};
    use crate::arg::search::ArgGraph;
    use petgraph::stable_graph::StableGraph;

    fn diamond_type() -> ModalType {
        ModalType::functor(ModalMode::Diamond, TypeCategory::Scene, 1, true)
    }

    #[test]
    fn coherent_triangle_passes() {
        // u -◇→ v -◇→ w, u -◇→ w: types should cohere after ◇ application
        let mut g: ArgGraph = StableGraph::new();
        let u_type = diamond_type();
        let applied = u_type.apply().unwrap();

        let mut n_u = ArgNode::new(1, NodeType::Concept, u_type, (0,0));
        let mut n_v = ArgNode::new(2, NodeType::Concept, applied, (0,0));
        let mut n_w = ArgNode::new(3, NodeType::Concept, applied.apply().unwrap_or(applied), (0,0));
        n_u.atms_label = 0b111; n_v.atms_label = 0b111; n_w.atms_label = 0b111;

        let u = g.add_node(n_u);
        let v = g.add_node(n_v);
        let w = g.add_node(n_w);
        g.add_edge(u, v, ArgEdge::new(1, 1, 2, EdgeType::Composition, ModalMode::Diamond));
        g.add_edge(v, w, ArgEdge::new(2, 2, 3, EdgeType::Composition, ModalMode::Diamond));
        g.add_edge(u, w, ArgEdge::new(3, 1, 3, EdgeType::Composition, ModalMode::Diamond));

        let stalks = build_stalks(&g);
        let result = check_sheaf_coherence(&g, &stalks);
        // No violations on a coherent ◇-composed triangle
        assert!(result.is_coherent(), "violations: {:?}", result.violations);
    }
}
