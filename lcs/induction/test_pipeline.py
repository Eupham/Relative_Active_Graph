import sys
from pathlib import Path
from ud_parser import UdParser
from ud_to_mtlg import ud_tree_to_mtlg
from morphological_fst import preprocess_for_type_assignment
from mc4_stream import stream_mc4
from mtlg_inducer import MtlgInducer
from trd_bootstrap import TrdBootstrapper, graph_to_modal_vector

def main():
    language = "en"
    max_samples = 50
    print(f"Streaming {max_samples} sentences from mC4 ({language})...")

    parser = UdParser(language)
    graphs = []

    for item in stream_mc4(language, max_samples=max_samples):
        try:
            tokens = preprocess_for_type_assignment(item["text"], language)
            sentence = " ".join(t.split("[")[0] for t in tokens)
            tree = parser.parse(sentence)
            graph = ud_tree_to_mtlg(tree)
            if graph.nodes:
                graphs.append(graph)
        except Exception as e:
            print(f"Error parsing sentence: {e}")
            pass

    print(f"Successfully created {len(graphs)} MTLG graphs.")

    print("Inducing MTLG lexicon...")
    inducer = MtlgInducer(language)
    lexicon = inducer.induce_from_stream(iter(graphs), max_trees=len(graphs))
    print(f"Lexicon created with {len(lexicon.entries)} lemmas.")

    print("Bootstrapping TRDs...")
    bootstrapper = TrdBootstrapper(n_clusters=4)
    trds = bootstrapper.bootstrap(graphs, language)
    print(f"Bootstrapped {len(trds)} TRDs.")

    print("\n--- Inference Test ---")
    test_sentence = "The scientist discovered a new particle in the laboratory."
    print(f"Analyzing test sentence: '{test_sentence}'")

    tokens = preprocess_for_type_assignment(test_sentence, language)
    clean = " ".join(t.split("[")[0] for t in tokens)
    tree = parser.parse(clean)
    graph = ud_tree_to_mtlg(tree)

    print(f"Graph nodes: {len(graph.nodes)}")
    print(f"Graph edges: {len(graph.edges)}")

    # Lexicon lookup per node
    for node in graph.nodes:
        entry = lexicon.best_type(node.lemma)
        lex_mode = entry.modal_mode if entry else "—"
        lex_cat = entry.ucca_cat if entry else "—"
        print(f"  Token: {node.text:<12} Lemma: {node.lemma:<12} Lex Mode: {lex_mode:<8} Lex Cat: {lex_cat}")

    # TRD assignment
    trd_label = "unknown"
    if bootstrapper._model and bootstrapper.vocab:
        vec = graph_to_modal_vector(graph, bootstrapper.vocab)
        trd_id = bootstrapper._model.predict(vec)
        for t in bootstrapper.trds:
            if t.trd_id == trd_id:
                trd_label = t.label
                break

    print(f"\nAssigned TRD cluster: {trd_label}")
    print("End-to-end pipeline test completed successfully.")

if __name__ == "__main__":
    main()
