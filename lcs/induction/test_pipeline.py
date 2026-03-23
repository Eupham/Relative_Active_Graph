"""Integration test — no UD, no Stanza."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import tokenize
from text_to_graph import text_to_mtlg
from mtlg_inducer import MtlgInducer
from mc4_stream import stream_mc4


def main():
    language    = "en"
    max_samples = 200
    print(f"Streaming {max_samples} sentences from mC4 ({language})...")
    graphs = []
    for item in stream_mc4(language, max_samples=max_samples):
        try:
            g = text_to_mtlg(item["text"], language)
            if g.nodes:
                graphs.append(g)
        except Exception:
            pass
    print(f"Created {len(graphs)} graphs.")
    inducer = MtlgInducer(language)
    inducer.induce_from_stream(iter(graphs), max_trees=len(graphs))
    print(f"Lexicon: {len(inducer.entries)} lemmas.")
    test_sentence = "The scientist discovered a new particle."
    g = text_to_mtlg(test_sentence, language)
    print(f"\nTest: '{test_sentence}'")
    print(f"Nodes: {len(g.nodes)}  Edges: {len(g.edges)}")
    for node in g.nodes:
        entry = inducer.best_type(node.lemma)
        cat   = entry.ucca_cat if entry else "—"
        print(f"  {node.text:<15} lemma={node.lemma:<15} cat={cat}")
    print("\nPipeline test passed.")


if __name__ == "__main__":
    main()
