//! CSRRE CLI: newline-delimited JSON protocol.
//! Handles: query, register_lexicon, execute_passage, shutdown.

use std::io::{self, BufRead, Write};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use csrre_core::{
    Engine, Query, Quality,
    arg::{ArgNode, NodeClass, ArgEdge, EdgeClass},
    types::{ModalType, ModalMode, TypeCategory, Direction, TRDId},
    engine::TokenStep,
    lcs::token_types::TokenStructure,
};

// ─── Wire types ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(tag = "type")]
enum WireMessage {
    #[serde(rename = "query")]
    Query(WireQuery),
    #[serde(rename = "register_lexicon")]
    RegisterLexicon(WireLexEntry),
    #[serde(rename = "execute_passage")]
    ExecutePassage(WirePassage),
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

#[derive(Deserialize, Clone)]
struct WireNode {
    id:           u64,
    surface:      Option<String>,
    score:        f32,
    mode:         Option<String>,
    cat:          Option<u32>,
    arity:        Option<u8>,
    suffix3_hash: Option<u64>,
    prefix2_hash: Option<u64>,
    suffix2_hash: Option<u64>,
}

#[derive(Deserialize, Clone)]
struct WireEdge {
    id:     u64,
    src:    u64,
    dst:    u64,
    mode:   Option<String>,
    weight: Option<f32>,
}

#[derive(Deserialize)]
struct WireLexEntry {
    predicate: String,
    language:  String,
    surface:   String,
}

#[derive(Deserialize)]
struct WirePassage {
    trd:       u32,
    #[serde(default = "default_lang")]
    language:  String,
    sentences: Vec<WireSentence>,
}

#[derive(Deserialize)]
struct WireSentence {
    #[serde(default)]
    nodes: Vec<WireNode>,
    #[serde(default)]
    edges: Vec<WireEdge>,
    #[serde(default)]
    steps: Vec<u64>,
}

#[derive(Serialize)]
struct WireResult {
    surface:   String,
    satisfied: bool,
    depth:     usize,
    quality:   f32,
}

#[derive(Serialize)]
struct WireTrainResult {
    steps:        usize,
    quality:      f32,
    mean_quality: f64,
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

    // Build TokenStructure from wire fields for online category induction.
    use std::collections::BTreeSet;
    let structure = TokenStructure {
        token_id:              w.id as u32,
        is_first_token:        false,
        is_last_token:         false,
        normalized_position:   0.5,
        sentence_length_norm:  0.5,
        starts_with_uppercase: w.surface.as_deref()
            .and_then(|s| s.chars().next())
            .map(|c| c.is_uppercase())
            .unwrap_or(false),
        is_punctuation:        false,
        is_repeated:           false,
        char_length_norm:      w.surface.as_deref()
            .map(|s| (s.len() as f32 / 15.0).min(1.0))
            .unwrap_or(0.3),
        prefix2_hash:          w.prefix2_hash.unwrap_or(0) as u32,
        suffix3_hash:          w.suffix3_hash.unwrap_or(0) as u32,
        suffix2_hash:          w.suffix2_hash.unwrap_or(0) as u32,
        prev_lemma_hash:       0,
        next_lemma_hash:       0,
        n_context_neighbors:   0,
        char_trigram_hashes:   BTreeSet::new(),
    };

    let mut node = ArgNode::new(w.id, NodeClass::DEFAULT, mt, (0, 0));
    if let Some(s) = w.surface { node.surface = Some(s.into_bytes()); }
    node.attribution_score = w.score;
    node.atms_label = 0b1;
    node.structure = Some(structure);
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

    log::info!("CSRRE engine ready.");

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
                let key = format!("{}:{}", entry.language, entry.predicate);
                engine.global_lexicon.insert(key, csrre_core::generation::linearizer::LexEntry {
                    predicate:  entry.predicate,
                    language:   entry.language,
                    surface:    entry.surface,
                    modal_type: ModalType::default(),
                });
                // No response for register_lexicon — fire and forget.
            }
            Ok(WireMessage::ExecutePassage(wp)) => {
                let sentences: Vec<Vec<TokenStep>> = wp.sentences.iter().map(|sent| {
                    let node_pool: Vec<ArgNode> = sent.nodes.iter().cloned().map(wire_node_to_arg).collect();
                    let edge_pool: Vec<ArgEdge> = sent.edges.iter().cloned().map(wire_edge_to_arg).collect();
                    sent.steps.iter().map(|&expected_node_id| TokenStep {
                        text:             String::new(),
                        expected_node_id,
                        node_pool:        node_pool.clone(),
                        edge_pool:        edge_pool.clone(),
                    }).collect::<Vec<_>>()
                }).collect();

                let result = engine.execute_passage(wp.trd, sentences, &wp.language);
                let wire = WireTrainResult {
                    steps:        result.steps_processed,
                    quality:      result.final_quality.as_f32(),
                    mean_quality: if result.steps_processed > 0 {
                        result.quality_sum / result.steps_processed as f64
                    } else { 0.0 },
                };
                writeln!(out, "{}", serde_json::to_string(&wire)?)?;
                out.flush()?;
            }
            Ok(WireMessage::Shutdown) => {
                log::info!("Shutdown received.");
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
