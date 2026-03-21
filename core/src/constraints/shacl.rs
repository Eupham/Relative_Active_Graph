//! SHACL-inspired structural validation over G(s).
//! Validates that ARG subgraphs satisfy declared shape constraints.
//! Shapes are defined as rules over node types, edge types, and modal modes.

use std::collections::HashMap;
use crate::types::{NodeId, ModalMode};
use crate::arg::{ArgGraph, node::NodeType, edge::EdgeType};
use petgraph::visit::EdgeRef;

/// A shape constraint that ARG structures must satisfy.
#[derive(Clone, Debug)]
pub struct ShapeConstraint {
    pub name: String,
    pub kind: ConstraintKind,
}

#[derive(Clone, Debug)]
pub enum ConstraintKind {
    /// Every node of `node_type` must have at least `min` outgoing edges of `edge_type`.
    MinOutgoing { node_type: NodeType, edge_type: EdgeType, min: usize },
    /// Every ◇-mode edge must connect nodes of compatible categories.
    DiamondModeCompatibility,
    /// □-mode edges must form a DAG (no cycles through sharing).
    BoxModeAcyclic,
    /// Every root node (no incoming edges) must have an explicit modal type annotation.
    RootsMustBeTyped,
}

/// Result of validating one constraint.
#[derive(Debug)]
pub struct ValidationViolation {
    pub constraint: String,
    pub node_id:    Option<NodeId>,
    pub message:    String,
}

/// Validate a subgraph G(s) against a set of shape constraints.
pub fn validate_shapes(
    graph: &ArgGraph,
    constraints: &[ShapeConstraint],
) -> Vec<ValidationViolation> {
    let mut violations = Vec::new();
    for constraint in constraints {
        match &constraint.kind {
            ConstraintKind::MinOutgoing { node_type, edge_type, min } => {
                check_min_outgoing(graph, constraint.name.clone(), *node_type, *edge_type, *min, &mut violations);
            }
            ConstraintKind::DiamondModeCompatibility => {
                check_diamond_compatibility(graph, constraint.name.clone(), &mut violations);
            }
            ConstraintKind::BoxModeAcyclic => {
                check_box_acyclic(graph, constraint.name.clone(), &mut violations);
            }
            ConstraintKind::RootsMustBeTyped => {
                check_roots_typed(graph, constraint.name.clone(), &mut violations);
            }
        }
    }
    violations
}

fn check_min_outgoing(
    graph: &ArgGraph,
    name: String,
    required_node_type: NodeType,
    required_edge_type: EdgeType,
    min: usize,
    violations: &mut Vec<ValidationViolation>,
) {
    for idx in graph.node_indices() {
        let node = &graph[idx];
        if node.node_type != required_node_type { continue; }
        let count = graph.edges(idx)
            .filter(|e| e.weight().edge_type == required_edge_type)
            .count();
        if count < min {
            violations.push(ValidationViolation {
                constraint: name.clone(),
                node_id:    Some(node.id),
                message:    format!("node {} has {} {:?} edges, need ≥ {}", node.id, count, required_edge_type, min),
            });
        }
    }
}

fn check_diamond_compatibility(
    graph: &ArgGraph,
    name: String,
    violations: &mut Vec<ValidationViolation>,
) {
    for idx in graph.node_indices() {
        for e in graph.edges(idx) {
            if e.weight().modal_mode != ModalMode::Diamond { continue; }
            let src_cat = graph[e.source()].mtlg_type.category;
            let dst_cat = graph[e.target()].mtlg_type.category;
            // ◇-edges should connect functor (non-saturated) to argument.
            let src_arity = graph[e.source()].mtlg_type.arity;
            if src_arity == 0 {
                violations.push(ValidationViolation {
                    constraint: name.clone(),
                    node_id:    Some(graph[e.source()].id),
                    message:    format!("◇-edge from saturated node {:?}→{:?}", src_cat, dst_cat),
                });
            }
        }
    }
}

fn check_box_acyclic(
    graph: &ArgGraph,
    name: String,
    violations: &mut Vec<ValidationViolation>,
) {
    // Simple cycle detection via DFS on □-mode edges only.
    use std::collections::HashSet;
    let mut visited   = HashSet::new();
    let mut rec_stack = HashSet::new();

    for start in graph.node_indices() {
        if !visited.contains(&start) {
            if has_box_cycle(graph, start, &mut visited, &mut rec_stack) {
                violations.push(ValidationViolation {
                    constraint: name.clone(),
                    node_id:    Some(graph[start].id),
                    message:    "□-mode cycle detected".into(),
                });
                return; // report once
            }
        }
    }
}

fn has_box_cycle(
    graph: &ArgGraph,
    node: petgraph::stable_graph::NodeIndex,
    visited: &mut std::collections::HashSet<petgraph::stable_graph::NodeIndex>,
    rec_stack: &mut std::collections::HashSet<petgraph::stable_graph::NodeIndex>,
) -> bool {
    visited.insert(node);
    rec_stack.insert(node);
    for e in graph.edges(node) {
        if e.weight().modal_mode != ModalMode::Box { continue; }
        let next = e.target();
        if !visited.contains(&next) {
            if has_box_cycle(graph, next, visited, rec_stack) { return true; }
        } else if rec_stack.contains(&next) {
            return true;
        }
    }
    rec_stack.remove(&node);
    false
}

fn check_roots_typed(
    graph: &ArgGraph,
    name: String,
    violations: &mut Vec<ValidationViolation>,
) {
    // A root has no incoming edges (in the directed graph).
    let has_incoming: std::collections::HashSet<_> = graph.edge_indices()
        .map(|ei| graph.edge_endpoints(ei).unwrap().1)
        .collect();

    for idx in graph.node_indices() {
        if !has_incoming.contains(&idx) {
            let node = &graph[idx];
            // "Typed" means arity > 0 or the surface is not empty.
            if node.mtlg_type.arity == 0 && node.surface.is_none() {
                violations.push(ValidationViolation {
                    constraint: name.clone(),
                    node_id:    Some(node.id),
                    message:    format!("root node {} lacks modal type annotation", node.id),
                });
            }
        }
    }
}

/// Build a standard set of constraints for a typical ARG.
pub fn default_constraints() -> Vec<ShapeConstraint> {
    vec![
        ShapeConstraint {
            name: "diamond-compat".into(),
            kind: ConstraintKind::DiamondModeCompatibility,
        },
        ShapeConstraint {
            name: "box-acyclic".into(),
            kind: ConstraintKind::BoxModeAcyclic,
        },
        ShapeConstraint {
            name: "roots-typed".into(),
            kind: ConstraintKind::RootsMustBeTyped,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg::{ArgNode, NodeType, ArgEdge, EdgeType};
    use petgraph::stable_graph::StableGraph;
    use crate::types::{ModalType, ModalMode, TypeCategory};

    #[test]
    fn valid_graph_no_violations() {
        let mut g: ArgGraph = StableGraph::new();
        let functor_type = ModalType::functor(ModalMode::Diamond, TypeCategory::Scene, 1, true);
        let mut n = ArgNode::new(1, NodeType::Concept, functor_type, (0, 0));
        n.surface = Some(b"test".to_vec());
        g.add_node(n);
        let violations = validate_shapes(&g, &default_constraints());
        // A single node with surface and functor type: only RootsMustBeTyped might fire.
        // arity=1 > 0, so it passes.
        assert!(violations.iter().all(|v| v.constraint != "roots-typed"),
            "unexpected violations: {:?}", violations);
    }
}
