"""Integration test for the UTF-8 character-level pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from mc4_stream import stream_mc4, _split_sentences
from c4_sequence_extractor import _sentence_to_sequence, stream_c4_sequences
from trd_bootstrap import TrdBootstrapper
from mtlg_inducer import MtlgInducer


def main():
    language = "en"
    max_samples = 200
    print(f"Streaming {max_samples} sentences from mC4 ({language})...")

    sentences = []
    for item in stream_mc4(language, max_samples=max_samples):
        for s in _split_sentences(item.get("text", "")):
            sentences.append(s)
    print(f"Collected {len(sentences)} sentences.")

    print("Inducing character-level MTLG lexicon...")
    inducer = MtlgInducer(language)
    for sentence in sentences:
        seq = _sentence_to_sequence(sentence, trd_id=0)
        for step in seq.steps:
            inducer.lexicon.update(step.lemma, "diamond", 0, 0, 1.0)
    inducer.lexicon.normalize()
    print(f"Lexicon: {len(inducer.lexicon.entries)} characters.")

    print("Bootstrapping TRDs...")
    bootstrapper = TrdBootstrapper(n_clusters=16)
    trds = bootstrapper.bootstrap(sentences, language)
    print(f"Bootstrapped {len(trds)} TRDs.")

    print("\n--- Inference Test ---")
    test = "The scientist discovered a new particle in the laboratory."
    seq  = _sentence_to_sequence(test, trd_id=0)
    print(f"Characters: {len(seq.steps)}")
    print("Sample node_ids (first 5):")
    for step in seq.steps[:5]:
        assert step.node_id == step.expected_node_id, \
            f"BUG: node_id {step.node_id} != expected_node_id {step.expected_node_id}"
        print(f"  {repr(step.text)} -> node_id={step.node_id:#014x}")

    print("\nPipeline test complete.")


if __name__ == "__main__":
    main()
