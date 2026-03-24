//! Context stack: push/pop/shift with integrated ATMS env management and TR lifecycle.
//! DRS referents lift to parent context on pop().

use crate::types::{ContextId, Env, TRDId, ModalType};
use crate::atms::ContextBridge;
use crate::arg::{
    transient_repr::{Tr, Lifecycle, RepContent, Granularity},
    lifting::LiftingRuleRegistry,
};

/// A predication condition in a DRS with accessibility tracking.
#[derive(Clone, Debug, PartialEq)]
pub struct DrsCondition {
    pub predicate:    String,
    pub args:         Vec<String>,
    /// Depth at which this condition originated. 0 = top-level, n = lifted from depth n.
    pub origin_depth: usize,
}

/// Discourse Referent Set accumulated within a context.
#[derive(Clone, Debug, Default)]
pub struct Drs {
    /// Variables introduced by TRs in this context.
    pub referents:  Vec<String>,
    /// Predication conditions with origin depth (Kamp & Reyle 1993 §1.3).
    pub conditions: Vec<DrsCondition>,
}

impl Drs {
    pub fn extend_from_tr(&mut self, content: &RepContent) {
        self.extend_from_tr_at_depth(content, 0);
    }

    pub fn extend_from_tr_at_depth(&mut self, content: &RepContent, depth: usize) {
        if let RepContent::DrsUpdate { referents, conditions } = content {
            for r in referents { if !self.referents.contains(r) { self.referents.push(r.clone()); } }
            for c in conditions {
                let cond = DrsCondition { predicate: c.clone(), args: vec![], origin_depth: depth };
                if !self.conditions.contains(&cond) { self.conditions.push(cond); }
            }
        }
    }

    /// Lift referents AND conditions to a parent DRS (on context pop).
    /// Per Kamp & Reyle 1993 §1.3: both referents and conditions must be accessible
    /// from a parent DRS.
    pub fn lift_to(&self, parent: &mut Drs) {
        for r in &self.referents {
            if !parent.referents.contains(r) { parent.referents.push(r.clone()); }
        }
        // Conditions MUST be lifted. Accessibility is the parent's responsibility.
        // Per Kamp & Reyle 1993 §1.3.
        for cond in &self.conditions {
            if !parent.conditions.contains(cond) {
                parent.conditions.push(cond.clone());
            }
        }
    }
}

/// A single entry on the context stack.
pub struct ContextFrame {
    pub id:          ContextId,
    pub situation_id: u64,
    pub atms_env:    Env,
    pub trd:         Option<TRDId>,
    pub active_trs:  Vec<Tr>,
    pub drs:         Drs,
    pub parent:      Option<ContextId>,
}

impl ContextFrame {
    pub fn new(id: ContextId, situation_id: u64, atms_env: Env, trd: Option<TRDId>, parent: Option<ContextId>) -> Self {
        Self {
            id, situation_id, atms_env, trd,
            active_trs: Vec::new(),
            drs:        Drs::default(),
            parent,
        }
    }
}

/// The full context stack with integrated ATMS bridge, DRS, and TR lifecycle.
pub struct ContextStack {
    frames:   Vec<ContextFrame>,
    pub bridge:   ContextBridge,
    pub lifting:  LiftingRuleRegistry,
    next_ctx_id:  ContextId,
    next_tr_id:   u64,
}

impl ContextStack {
    pub fn new() -> Self {
        Self {
            frames:      Vec::new(),
            bridge:      ContextBridge::new(),
            lifting:     LiftingRuleRegistry::new(),
            next_ctx_id: 0,
            next_tr_id:  0,
        }
    }

    /// Push: activate A(s); look up TRD; build G(s) lazily (search module does this).
    /// Returns ContextId. On AtmsError::AssumptionSpaceExhausted, logs and uses env=0 fallback.
    pub fn push(&mut self, situation_id: u64, trd: Option<TRDId>) -> ContextId {
        let id = self.next_ctx_id;
        self.next_ctx_id += 1;
        let parent = self.frames.last().map(|f| f.id);
        let env = match self.bridge.push(id) {
            Ok(e) => e,
            Err(e) => {
                log::error!("ContextStack::push: {e}; using env=0 fallback");
                0
            }
        };
        self.frames.push(ContextFrame::new(id, situation_id, env, trd, parent));
        id
    }

    /// Pop: dissolve TRs, lift DRS to parent, deactivate ATMS env O(1).
    /// Returns (dissolved_trs, lifted_referents).
    pub fn pop(&mut self) -> Option<(Vec<Tr>, Vec<String>)> {
        let (ctx_id, _bit, _env) = self.bridge.pop()?;
        let frame = self.frames.pop()?;
        debug_assert_eq!(frame.id, ctx_id);

        // Lift DRS referents to parent.
        let lifted_referents = frame.drs.referents.clone();
        if let Some(parent) = self.frames.last_mut() {
            frame.drs.lift_to(&mut parent.drs);
        }

        // Dissolve all active TRs.
        let dissolved = frame.active_trs.into_iter().map(|mut tr| {
            tr.begin_dissolve();
            tr.complete_dissolve();
            tr
        }).collect();

        Some((dissolved, lifted_referents))
    }

    /// Shift: lift TRs with matching lifting rules; dissolve others.
    pub fn shift(&mut self, to_situation_id: u64, trd: Option<TRDId>) -> ContextId {
        let old_id = self.frames.last().map(|f| f.id).unwrap_or(0);
        // pop old context
        let (mut dissolved_trs, referents) = self.pop().unwrap_or_default();
        // Reclaim pending bits so push() can reuse the bit freed by pop().
        // Without this, the bit sits in pending_free and push() must allocate
        // a fresh bit, exhausting the 64-bit assumption space after 64 shifts.
        self.bridge.try_reclaim_pending(&[]);
        // push new context
        let new_id = self.push(to_situation_id, trd);
        // re-lift TRs that have matching rules
        let new_frame = self.frames.last_mut().unwrap();
        for mut tr in dissolved_trs.drain(..) {
            if let Some(new_type) = self.lifting.lift_tr(&tr, old_id, new_id) {
                tr.lifecycle = Lifecycle::Active;
                tr.mtlg_type = new_type;
                new_frame.active_trs.push(tr);
            }
        }
        for r in referents { if !new_frame.drs.referents.contains(&r) { new_frame.drs.referents.push(r); } }
        new_id
    }

    /// Add a TR to the current context.
    pub fn add_tr(&mut self, content: RepContent, env: Env, mtlg_type: ModalType, granularity: Granularity) -> u64 {
        let id = self.next_tr_id;
        self.next_tr_id += 1;
        let ctx_id = self.frames.last().map(|f| f.id).unwrap_or(0);
        let depth  = self.frames.len();
        let tr = Tr::new(id, ctx_id, content.clone(), env, mtlg_type, granularity);
        if let Some(frame) = self.frames.last_mut() {
            frame.drs.extend_from_tr_at_depth(&content, depth);
            frame.active_trs.push(tr);
        }
        id
    }

    pub fn current_env(&self) -> Env { self.bridge.active_env() }
    pub fn current_trd(&self) -> Option<TRDId> { self.frames.last().and_then(|f| f.trd) }
    pub fn depth(&self) -> usize { self.frames.len() }
    pub fn current_drs(&self) -> Option<&Drs> { self.frames.last().map(|f| &f.drs) }
}

impl Default for ContextStack {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_drs_lifts() {
        let mut stack = ContextStack::new();
        stack.push(1, None);
        stack.add_tr(
            RepContent::DrsUpdate { referents: vec!["x".into()], conditions: vec![] },
            0b01,
            crate::types::ModalType::default(),
            Granularity::Discourse,
        );
        let (_, refs) = stack.pop().unwrap();
        assert!(refs.contains(&"x".into()));
    }
}
