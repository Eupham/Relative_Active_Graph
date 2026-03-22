//! CSRRE CLI: reads JSON queries from stdin, writes JSON results to stdout.
//! Protocol: newline-delimited JSON. Each line is one query or control message.
//!
//! Query format:
//!   {"type":"query","text":"...","situation_id":1,"trd":null,"language":"en","nodes":[...],"edges":[...]}
//! Result format:
//!   {"surface":"...","satisfied":true,"depth":0,"quality":0.9}
//!
//! Train format:
//!   {"type":"train_sequence","trd":0,"tokens":[{"text":"...","expected_edge_id":1},...],"language":"en"}

use std::io::{self, BufRead, Write};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use csrre_core::{
    Engine, Query, Quality,
    arg::{ArgNode, NodeClass, ArgEdge, EdgeClass},
    types::{ModalType, ModalMode, TypeCategory, Direction, TRDId},
};

// ─── Wire protocol types ──────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(tag = "type")]
enum WireMessage {
    #[serde(rename = "query")]
    Query(WireQuery),
    #[serde(rename = "register_lexicon")]
    RegisterLexicon(WireLexEntry),
    #[serde(rename = "shutdown")]
    Shutdown,
}

#[derive(Deserialize)]
struct WireQuery {
    text:         String,
    situation_id: u64,
    trd:          Option<TRDId>,
    #[serde(default = "default_lang")]
    language:     String,
    #[serde(default)]
    nodes:        Vec<WireNode>,
    #[serde(default)]
    edges:        Vec<WireEdge>,
}

#[derive(Deserialize)]
struct WireNode {
    id:      u64,
    surface: Option<String>,
    score:   f32,
    mode:    Option<String>,   // "diamond", "box", "lozenge"
    /// Numeric category ID (0 = DEFAULT/unassigned, 1-6 = structural prototypes).
    cat:     Option<u32>,
    arity:   Option<u8>,
}

#[derive(Deserialize)]
struct WireEdge {
    id:   u64,
    src:  u64,
    dst:  u64,
    mode: Option<String>,
    weight: Option<f32>,
}

#[derive(Deserialize)]
struct WireLexEntry {
    predicate:  String,
    language:   String,
    surface:    String,
    role_order: Option<Vec<String>>,  // new optional field; None = []
}

#[derive(Serialize)]
struct WireResult {
    surface:   String,
    satisfied: bool,
    depth:     usize,
    /// Quality as a float in [0.0, 1.0].
    quality:   f32,
}

fn default_lang() -> String { "en".into() }

// ─── Conversion helpers ───────────────────────────────────────────────────────

fn parse_mode(s: Option<&str>) -> ModalMode {
    match s {
        Some("box")     => ModalMode::Box,
        Some("lozenge") => ModalMode::Lozenge,
        _               => ModalMode::Diamond,
    }
}

fn wire_node_to_arg(w: WireNode) -> ArgNode {
    let mode  = parse_mode(w.mode.as_deref());
    let cat   = TypeCategory(w.cat.unwrap_or(0));
    let arity = w.arity.unwrap_or(0);
    let mt    = if arity > 0 {
        ModalType::functor(mode, cat, arity, Direction::Right)
    } else {
        ModalType::atom(mode, cat)
    };
    let mut node = ArgNode::new(w.id, NodeClass::DEFAULT, mt, (0, 0));
    if let Some(s) = w.surface { node.surface = Some(s.into_bytes()); }
    node.attribution_score = w.score;
    node.atms_label = 0b1; // all nodes active in first context
    node
}

fn wire_edge_to_arg(w: WireEdge) -> ArgEdge {
    let mode = parse_mode(w.mode.as_deref());
    let mut edge = ArgEdge::new(w.id, w.src, w.dst, EdgeClass::DEFAULT, mode);
    if let Some(wt) = w.weight { edge.weight = wt; }
    edge
}

// ─── Main loop ────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    env_logger::init();
    let mut engine = Engine::new();
    let stdin  = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    log::info!("CSRRE engine ready. Reading newline-delimited JSON from stdin.");

    for line in stdin.lock().lines() {
        let line = line.context("failed to read stdin")?;
        if line.trim().is_empty() { continue; }

        match serde_json::from_str::<WireMessage>(&line) {
            Ok(WireMessage::Query(wq)) => {
                let nodes: Vec<ArgNode> = wq.nodes.into_iter().map(wire_node_to_arg).collect();
                let edges: Vec<ArgEdge> = wq.edges.into_iter().map(wire_edge_to_arg).collect();
                let query = Query {
                    text:            wq.text,
                    situation_id:    wq.situation_id,
                    trd:             wq.trd,
                    target_language: wq.language,
                    expected_type:   ModalType::default(),
                };
                let result = engine.execute(query, nodes, edges);
                let wire = WireResult {
                    surface:   result.surface_output,
                    satisfied: result.satisfied,
                    depth:     result.depth_used,
                    quality:   result.quality.as_f32(),
                };
                writeln!(out, "{}", serde_json::to_string(&wire)?)?;
                out.flush()?;
            }
            Ok(WireMessage::RegisterLexicon(entry)) => {
                log::info!("Registered lexicon: {} → {} ({})", entry.predicate, entry.surface, entry.language);
                let key = format!("{}:{}", entry.language, entry.predicate);
                engine.global_lexicon.insert(key, csrre_core::generation::LexEntry {
                    predicate:  entry.predicate,
                    language:   entry.language,
                    surface:    entry.surface,
                    modal_type: ModalType::default(),
                    role_order: entry.role_order.unwrap_or_default(),
                });
            }
            Ok(WireMessage::Shutdown) => {
                log::info!("Received shutdown.");
                break;
            }
            Err(e) => {
                log::error!("Parse error: {}: {}", e, line);
                let err = serde_json::json!({"error": e.to_string()});
                writeln!(out, "{}", err)?;
                out.flush()?;
            }
        }
    }
    Ok(())
}
