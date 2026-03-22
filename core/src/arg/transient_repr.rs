//! Transient Representation (TR) — DOLCE Perdurant, scoped to a context.
//! ist(c, R) holds ↔ τ = active. On pop(c) or NOGOOD → τ = dissolving → dissolved.
//! The ARG is modified ONLY by dissolved TR attribution traces.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::types::{NodeId, EdgeId, ContextId, ModalType, Env};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lifecycle {
    Active,
    Dissolving,
    Dissolved,
}

/// Granularity of a TR's semantic content.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Granularity {
    /// Sentence-level: λ-term → AMR-style proposition graph.
    Sentence,
    /// Discourse-level: DRS update (variable binding across TRs).
    Discourse,
    /// Passage-level: spans multiple sentences; attribution deferred until passage end.
    /// The context frame stays open across sentence boundaries and only dissolves
    /// (with backward attribution propagation) when the full passage is complete.
    Passage,
}

/// The semantic content of a TR.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RepContent {
    /// A lambda expression (string-serialized for now; would be LambdaTerm in full system).
    Lambda(String),
    /// A DRS update: new referents + conditions.
    DrsUpdate { referents: Vec<String>, conditions: Vec<String> },
    /// A structured output (code, query, etc.).
    Structured(serde_json::Value),
}

/// Attribution trace: EdgeId → delta score contribution for this TR.
pub type AttributionTrace = HashMap<EdgeId, f32>;

/// A Transient Representation.
#[derive(Clone, Debug)]
pub struct Tr {
    pub id:               u64,
    pub context_id:       ContextId,
    pub content:          RepContent,
    pub lifecycle:        Lifecycle,
    pub atms_env:         Env,
    pub attribution_trace: AttributionTrace,
    /// Nodes whose derivations contributed to this TR.
    pub provenance:       Vec<NodeId>,
    pub mtlg_type:        ModalType,
    pub granularity:      Granularity,
}

impl Tr {
    pub fn new(
        id: u64,
        context_id: ContextId,
        content: RepContent,
        atms_env: Env,
        mtlg_type: ModalType,
        granularity: Granularity,
    ) -> Self {
        Self {
            id,
            context_id,
            content,
            lifecycle:        Lifecycle::Active,
            atms_env,
            attribution_trace: HashMap::new(),
            provenance:        Vec::new(),
            mtlg_type,
            granularity,
        }
    }

    /// Begin dissolution. Returns the attribution trace to apply to the ARG.
    pub fn begin_dissolve(&mut self) -> &AttributionTrace {
        self.lifecycle = Lifecycle::Dissolving;
        &self.attribution_trace
    }

    /// Complete dissolution. After this, ist(c, R) no longer holds.
    pub fn complete_dissolve(&mut self) {
        self.lifecycle = Lifecycle::Dissolved;
    }

    pub fn is_active(&self) -> bool {
        self.lifecycle == Lifecycle::Active
    }

    pub fn record_edge_contribution(&mut self, edge_id: EdgeId, delta: f32) {
        *self.attribution_trace.entry(edge_id).or_insert(0.0) += delta;
    }

    pub fn add_provenance(&mut self, node_id: NodeId) {
        if !self.provenance.contains(&node_id) {
            self.provenance.push(node_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModalMode, TypeCategory};

    fn make_tr() -> Tr {
        use crate::types::ModalType;
        Tr::new(
            1, 100,
            RepContent::Lambda("λx.pred(x)".into()),
            0b111,
            ModalType::atom(ModalMode::Diamond, TypeCategory::DEFAULT),
            Granularity::Sentence,
        )
    }

    #[test]
    fn lifecycle_transitions() {
        let mut tr = make_tr();
        assert!(tr.is_active());
        tr.begin_dissolve();
        assert_eq!(tr.lifecycle, Lifecycle::Dissolving);
        tr.complete_dissolve();
        assert_eq!(tr.lifecycle, Lifecycle::Dissolved);
    }

    #[test]
    fn attribution_accumulates() {
        let mut tr = make_tr();
        tr.record_edge_contribution(42, 0.3);
        tr.record_edge_contribution(42, 0.2);
        assert!((tr.attribution_trace[&42] - 0.5).abs() < 1e-6);
    }
}
