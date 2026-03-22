use csrre_core::generation::hypothesis::{Hypothesis, generate_hypotheses, filter_satisfying};
use csrre_core::generation::linearizer::{Linearizer, LexEntry, PerLanguageLexicon};
use csrre_core::semantics::mtlg_semantics::{MtlgSemantics, PropositionGraph};
use csrre_core::types::{ModalType, ModalMode, TypeCategory, Direction};
use csrre_core::arg::{ArgGraph, ArgNode, NodeClass};
use petgraph::stable_graph::StableGraph;
use std::collections::HashMap;

fn create_mock_node(id: u64, surface: &str, score: f32) -> ArgNode {
    let mut n = ArgNode::new(
        id,
        NodeClass::DEFAULT,
        ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right),
        (0, 0),
    );
    n.surface = Some(surface.as_bytes().to_vec());
    n.attribution_score = score;
    n.atms_label = 0b1;
    n
}

#[test]
fn test_generative_inference_pipeline() {
    // 1. Setup Semantic and Lexicon Context
    let sem = MtlgSemantics::new();
    let mut type_map = HashMap::new();
    type_map.insert(
        "run".into(),
        ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right),
    );
    type_map.insert(
        "alice".into(),
        ModalType::atom(ModalMode::Diamond, TypeCategory(6)),
    );

    let mut linearizer = Linearizer::new("en");
    linearizer.lexicon.register(LexEntry {
        predicate: "run".into(),
        language: "en".into(),
        surface: "runs".into(),
        modal_type: ModalType::functor(ModalMode::Diamond, TypeCategory::DEFAULT, 1, Direction::Right),
    });
    linearizer.lexicon.register(LexEntry {
        predicate: "alice".into(),
        language: "en".into(),
        surface: "Alice".into(),
        modal_type: ModalType::atom(ModalMode::Diamond, TypeCategory(6)),
    });

    // 2. Build mock ARG Graph
    let mut g: ArgGraph = StableGraph::new();
    let run_node = g.add_node(create_mock_node(1, "run", 0.9));
    let alice_node = g.add_node(create_mock_node(2, "alice", 0.8));

    // We can simulate hypothesis generation logic manually as `generate_hypotheses` requires `PerfRegistry`.
    // Since we're demonstrating the pipeline, let's manually build a Hypothesis like the search module would
    let prop = PropositionGraph {
        root: "run".into(),
        roles: vec![("ARG0".into(), "alice".into())],
        lambda_str: "run(alice)".into(),
    };

    let hypothesis = Hypothesis {
        id: 1,
        lambda_str: "run(alice)".into(),
        proposition: prop,
        relevance: 0.95,
        root_node: 1,
    };

    // 3. Test Type-Filtering (Satisfiability)
    let expected_type = ModalType::atom(ModalMode::Diamond, TypeCategory::DEFAULT);
    let satisfying_hyps = filter_satisfying(vec![hypothesis.clone()], &expected_type, &type_map);
    assert_eq!(satisfying_hyps.len(), 1, "Hypothesis must satisfy the query constraint");

    // 4. Test Linearization (Surface Generation)
    let output_str = linearizer.linearize(&satisfying_hyps[0]);
    assert_eq!(output_str, "runs Alice", "Linearizer must map logic predicates to their localized surface forms");

    println!("Generative Inference Integration Test Passed:");
    println!(" - Validated Hypothesis semantic mapping.");
    println!(" - Successfully localized 'run(alice)' to '{}'", output_str);
}
