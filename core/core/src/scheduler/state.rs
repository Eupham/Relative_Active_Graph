//! Node execution states: ACTIVE / BLOCKED / DORMANT.
//! ATMS-integrated: state transitions fire only when ATMS labels change.

use std::collections::HashMap;
use crate::types::{NodeId, Env};
use crate::atms::base::env::subsumes;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecState {
    /// Ready to execute: all dependencies satisfied in active env.
    Active,
    /// Waiting for one or more dependencies to be derived.
    Blocked,
    /// No applicable justification in current env; suspended until env changes.
    Dormant,
}

/// Execution state record for a node.
#[derive(Clone, Debug)]
pub struct NodeExecState {
    pub node_id:   NodeId,
    pub state:     ExecState,
    /// Required environment: the node can execute only if active_env ⊇ required_env.
    pub required_env: Env,
    /// Dependencies that must be ACTIVE before this node fires.
    pub depends_on: Vec<NodeId>,
    /// b/t-level (list-scheduling level for parallelism).
    pub bt_level:   usize,
}

impl NodeExecState {
    pub fn new(node_id: NodeId, required_env: Env, depends_on: Vec<NodeId>) -> Self {
        Self {
            node_id,
            state:        ExecState::Dormant,
            required_env,
            depends_on,
            bt_level:     0,
        }
    }

    /// Recompute state given the current ATMS active_env and the set of completed nodes.
    pub fn recompute(&mut self, active_env: Env, completed: &std::collections::HashSet<NodeId>) {
        if !subsumes(self.required_env, active_env) {
            self.state = ExecState::Dormant;
        } else if self.depends_on.iter().all(|dep| completed.contains(dep)) {
            self.state = ExecState::Active;
        } else {
            self.state = ExecState::Blocked;
        }
    }
}

/// Manages execution states for all nodes in the current G(s).
pub struct ExecStateTable {
    states:    HashMap<NodeId, NodeExecState>,
    completed: std::collections::HashSet<NodeId>,
    active_env: Env,
}

impl ExecStateTable {
    pub fn new(active_env: Env) -> Self {
        Self {
            states:    HashMap::new(),
            completed: std::collections::HashSet::new(),
            active_env,
        }
    }

    pub fn register(&mut self, state: NodeExecState) {
        self.states.insert(state.node_id, state);
    }

    /// Mark a node as completed and recompute downstream states.
    pub fn complete(&mut self, node_id: NodeId) {
        self.completed.insert(node_id);
        self.recompute_all();
    }

    pub fn recompute_all(&mut self) {
        let env = self.active_env;
        let completed = self.completed.clone();
        for s in self.states.values_mut() {
            s.recompute(env, &completed);
        }
    }

    pub fn update_env(&mut self, new_env: Env) {
        self.active_env = new_env;
        self.recompute_all();
    }

    /// Nodes ready to execute, sorted by b/t-level (ascending for list scheduling).
    pub fn ready_nodes(&self) -> Vec<NodeId> {
        let mut ready: Vec<_> = self.states.values()
            .filter(|s| s.state == ExecState::Active)
            .collect();
        ready.sort_by_key(|s| s.bt_level);
        ready.iter().map(|s| s.node_id).collect()
    }

    pub fn state_of(&self, node_id: NodeId) -> Option<ExecState> {
        self.states.get(&node_id).map(|s| s.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atms::base::env::singleton;

    #[test]
    fn active_when_env_and_deps_satisfied() {
        let env = singleton(0) | singleton(1);
        let mut table = ExecStateTable::new(env);
        table.register(NodeExecState::new(1, singleton(0), vec![]));
        table.register(NodeExecState::new(2, singleton(1), vec![1]));
        table.recompute_all();
        assert_eq!(table.state_of(1), Some(ExecState::Active));
        assert_eq!(table.state_of(2), Some(ExecState::Blocked)); // dep 1 not done yet

        table.complete(1);
        assert_eq!(table.state_of(2), Some(ExecState::Active));
    }

    #[test]
    fn dormant_when_env_unsatisfied() {
        let env = singleton(0);
        let mut table = ExecStateTable::new(env);
        table.register(NodeExecState::new(1, singleton(1), vec![])); // needs bit 1, active has bit 0
        table.recompute_all();
        assert_eq!(table.state_of(1), Some(ExecState::Dormant));
    }
}
