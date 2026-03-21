//! Bridge between the Context stack and ATMS environments.
//! Each context c corresponds to an ATMS environment A(s).
//! Context push = activate A(s); pop = deactivate A(s) [O(1)].

use std::collections::HashMap;
use crate::types::{ContextId, Env};
use crate::atms::base::env::{singleton, union, remove_bit, empty};

/// Manages the mapping from contexts to ATMS environments.
/// At most 64 simultaneously active contexts (one bit per context).
pub struct ContextBridge {
    /// Next available assumption bit.
    next_bit: u8,
    /// ContextId → assumption bit.
    context_to_bit: HashMap<ContextId, u8>,
    /// Currently active environment (union of all active context bits).
    active_env: Env,
    /// Stack of active context IDs (for pop semantics).
    stack: Vec<ContextId>,
}

impl ContextBridge {
    pub fn new() -> Self {
        Self {
            next_bit:       0,
            context_to_bit: HashMap::new(),
            active_env:     empty(),
            stack:          Vec::new(),
        }
    }

    /// Allocate a new assumption bit for `ctx` and activate it.
    /// Returns the resulting active environment.
    pub fn push(&mut self, ctx: ContextId) -> Env {
        assert!(self.next_bit < 64, "exceeded 64-context assumption space");
        let bit = self.next_bit;
        self.next_bit += 1;
        self.context_to_bit.insert(ctx, bit);
        self.active_env = union(self.active_env, singleton(bit));
        self.stack.push(ctx);
        self.active_env
    }

    /// Deactivate the current context's bit. O(1).
    /// Returns the bit that was deactivated (so BaseAtms can find affected nodes).
    pub fn pop(&mut self) -> Option<(ContextId, u8, Env)> {
        let ctx = self.stack.pop()?;
        let bit = *self.context_to_bit.get(&ctx)?;
        self.active_env = remove_bit(self.active_env, bit);
        Some((ctx, bit, self.active_env))
    }

    /// Shift: update assumptions by removing `remove_bits` and adding `add_ctx`.
    pub fn shift(&mut self, remove_ctxs: &[ContextId], add_ctx: Option<ContextId>) -> Env {
        for ctx in remove_ctxs {
            if let Some(&bit) = self.context_to_bit.get(ctx) {
                self.active_env = remove_bit(self.active_env, bit);
            }
        }
        if let Some(ctx) = add_ctx {
            let bit = self.next_bit;
            self.next_bit += 1;
            self.context_to_bit.insert(ctx, bit);
            self.active_env = union(self.active_env, singleton(bit));
            self.stack.push(ctx);
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
        let env1 = bridge.push(1);
        let env2 = bridge.push(2);
        assert_ne!(env1, env2);
        let (ctx, _bit, env_after_pop) = bridge.pop().unwrap();
        assert_eq!(ctx, 2);
        assert_eq!(env_after_pop, env1);
    }

    #[test]
    fn shift_updates_active() {
        let mut bridge = ContextBridge::new();
        bridge.push(1);
        bridge.push(2);
        let env = bridge.shift(&[2], Some(3));
        assert!(bridge.env_for(3).map_or(false, |e| e & env != 0));
    }
}
