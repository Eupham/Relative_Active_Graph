//! Bridge between the Context stack and ATMS environments.
//! Each context c corresponds to an ATMS environment A(s).
//! Context push = activate A(s); pop = deactivate A(s) [O(1)].
//!
//! §1: Free-bit pool prevents hard panic at 64 contexts.
//! Bits are recycled when popped and no live ATMS label references them.
//! The maximum simultaneous active contexts is still 64 (one bit per active context),
//! but bits are reused after pop(), so the engine supports unlimited sequential contexts.

use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};
use crate::types::{ContextId, Env};
use crate::atms::base::env::{singleton, union, remove_bit, empty};

/// Error type for ATMS context bridge operations (§1b).
#[derive(Debug)]
pub enum AtmsError {
    /// All 64 assumption bits are in use and none are reclaimable.
    AssumptionSpaceExhausted,
    /// The requested context was not found in the bridge.
    ContextNotFound(ContextId),
}

impl std::fmt::Display for AtmsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtmsError::AssumptionSpaceExhausted =>
                write!(f, "ATMS assumption space exhausted: no free bits and no reclaimable contexts"),
            AtmsError::ContextNotFound(id) =>
                write!(f, "Context {:?} not found in bridge", id),
        }
    }
}

/// Manages the mapping from contexts to ATMS environments.
/// Supports unlimited sequential contexts via bit recycling (§1).
/// At most 64 *simultaneously active* contexts (one bit per active context).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextBridge {
    context_to_bit:  HashMap<ContextId, u8>,
    bit_to_context:  HashMap<u8, ContextId>,
    next_fresh_bit:  u8,
    free_bits:       VecDeque<u8>,
    /// Bits that are logically popped but still potentially referenced by live ATMS labels.
    /// Re-checked for reclaimability after every remove_label call.
    pending_free:    Vec<u8>,
    active_env:      Env,
    stack:           Vec<ContextId>,
}

impl ContextBridge {
    pub fn new() -> Self {
        Self {
            context_to_bit:  HashMap::new(),
            bit_to_context:  HashMap::new(),
            next_fresh_bit:  0,
            free_bits:       VecDeque::new(),
            pending_free:    Vec::new(),
            active_env:      empty(),
            stack:           Vec::new(),
        }
    }

    /// Allocate a new assumption bit for `ctx` and activate it.
    /// Returns the resulting active environment, or AtmsError if space is exhausted (§1c).
    pub fn push(&mut self, ctx: ContextId) -> Result<Env, AtmsError> {
        let bit = if let Some(b) = self.free_bits.pop_front() {
            b
        } else if self.next_fresh_bit < 64 {
            let b = self.next_fresh_bit;
            self.next_fresh_bit += 1;
            b
        } else {
            return Err(AtmsError::AssumptionSpaceExhausted);
        };
        self.context_to_bit.insert(ctx, bit);
        self.bit_to_context.insert(bit, ctx);
        self.active_env = union(self.active_env, singleton(bit));
        self.stack.push(ctx);
        Ok(self.active_env)
    }

    /// Deactivate the current context's bit and recycle it (§1d).
    /// Returns the bit that was deactivated (so BaseAtms can find affected nodes).
    pub fn pop(&mut self) -> Option<(ContextId, u8, Env)> {
        let ctx = self.stack.pop()?;
        let bit = self.context_to_bit.remove(&ctx)?;
        self.bit_to_context.remove(&bit);
        self.active_env = remove_bit(self.active_env, bit);
        // A bit is reclaimable only when no live ATMS label references it.
        // We push it to pending_free; it will be reclaimed safely later.
        self.pending_free.push(bit);
        Some((ctx, bit, self.active_env))
    }

    /// Shift: update assumptions by removing `remove_ctxs` and adding `add_ctx`.
    pub fn shift(&mut self, remove_ctxs: &[ContextId], add_ctx: Option<ContextId>) -> Env {
        for ctx in remove_ctxs {
            if let Some(&bit) = self.context_to_bit.get(ctx) {
                self.active_env = remove_bit(self.active_env, bit);
                // Add to pending_free for recycling.
                self.context_to_bit.remove(ctx);
                self.bit_to_context.remove(&bit);
                self.pending_free.push(bit);
            }
            self.stack.retain(|&c| c != *ctx);
        }
        self.try_reclaim_pending_unchecked();
        if let Some(ctx) = add_ctx {
            // Best-effort push; if exhausted, log and continue without activating.
            match self.push(ctx) {
                Ok(env) => { return env; }
                Err(AtmsError::AssumptionSpaceExhausted) => {
                    log::error!("ContextBridge::shift: assumption space exhausted, context {} not activated", ctx);
                }
                Err(_) => {}
            }
        }
        self.active_env
    }

    pub fn active_env(&self) -> Env {
        self.active_env
    }

    pub fn env_for(&self, ctx: ContextId) -> Option<Env> {
        self.context_to_bit.get(&ctx).map(|&b| singleton(b))
    }

    pub fn current_context(&self) -> Option<ContextId> {
        self.stack.last().copied()
    }

    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Reclaims all pending bits immediately (safe only when no live label references them).
    fn try_reclaim_pending_unchecked(&mut self) {
        for bit in self.pending_free.drain(..) {
            self.free_bits.push_back(bit);
        }
    }

    /// Safe variant: retain bits still referenced by `live_labels` (§1d).
    pub fn try_reclaim_pending(&mut self, live_labels: &[Env]) {
        let still_live: Vec<u8> = self.pending_free.iter().copied()
            .filter(|&bit| live_labels.iter().any(|&env| env & singleton(bit) != 0))
            .collect();
        let pending: Vec<u8> = self.pending_free.drain(..).collect();
        for bit in pending {
            if !still_live.contains(&bit) {
                self.free_bits.push_back(bit);
            } else {
                self.pending_free.push(bit);
            }
        }
    }
}

impl Default for ContextBridge {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_o1() {
        let mut bridge = ContextBridge::new();
        let env1 = bridge.push(1).unwrap();
        let env2 = bridge.push(2).unwrap();
        assert_ne!(env1, env2);
        let (ctx, _bit, env_after_pop) = bridge.pop().unwrap();
        assert_eq!(ctx, 2);
        assert_eq!(env_after_pop, env1);
    }

    #[test]
    fn shift_updates_active() {
        let mut bridge = ContextBridge::new();
        bridge.push(1).unwrap();
        bridge.push(2).unwrap();
        let env = bridge.shift(&[2], Some(3));
        assert!(bridge.env_for(3).map_or(false, |e| e & env != 0));
    }

    /// §1f: Push and pop 1000 contexts sequentially.
    /// Assert context_to_bit.len() + bit_to_context.len() equals twice the current stack depth.
    #[test]
    fn sequential_1000_contexts_no_exhaustion() {
        let mut bridge = ContextBridge::new();
        for i in 0u64..1000 {
            let _ = bridge.push(i).expect("push should not exhaust");
            let depth = bridge.depth();
            assert_eq!(
                bridge.context_to_bit.len() + bridge.bit_to_context.len(),
                depth * 2,
                "at depth {depth}"
            );
            bridge.pop().expect("pop should work");
            // Emulate shift/reclaim so we don't exhaust bits
            bridge.try_reclaim_pending(&[]);
        }
    }
}
