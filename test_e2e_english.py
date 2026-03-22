import sys
import json
import subprocess
from pathlib import Path

# Add lcs/induction to path so we can import the parsers
sys.path.insert(0, str(Path("lcs/induction").absolute()))

from ud_parser import UdParser
from ud_to_mtlg import ud_tree_to_mtlg
from morphological_fst import preprocess_for_type_assignment

def build_cli_query(sentence: str, situation_id: int) -> dict:
    language = "en"

    # 1. Parse natural language into Universal Dependencies (UD)
    parser = UdParser(language)
    tokens = preprocess_for_type_assignment(sentence, language)
    clean = " ".join(t.split("[")[0] for t in tokens)
    tree = parser.parse(clean)

    # 2. Convert UD tree into MTLG semantic graph
    graph = ud_tree_to_mtlg(tree)

    # 3. Format as a wire message for the Rust engine
    wire_nodes = []
    for n in graph.nodes:
        # The Rust engine expects numeric category ID.
        wire_nodes.append({
            "id": n.token_id,
            "surface": n.text,
            "score": 0.9, # Mock high attribution
            "mode": n.modal_mode,
            "cat": n.category_id,
            "arity": n.arity
        })

    wire_edges = []
    for idx, e in enumerate(graph.edges):
        wire_edges.append({
            "id": idx + 1,
            "src": e.src_id,
            "dst": e.dst_id,
            "mode": e.modal_mode,
            "weight": 1.0
        })

    return {
        "type": "query",
        "text": sentence,
        "situation_id": situation_id,
        "language": language,
        "nodes": wire_nodes,
        "edges": wire_edges
    }

def main():
    if len(sys.argv) > 1:
        test_sentence = " ".join(sys.argv[1:])
    else:
        test_sentence = "Alice discovers a particle."

    print(f"--- Input (Plain English) ---\n{test_sentence}\n")

    # Generate the query payload using the python parsers
    query_payload = build_cli_query(test_sentence, 42)
    payload_str = json.dumps(query_payload)

    # Pre-load lexicon words needed
    words = []
    for node in query_payload["nodes"]:
        words.append(node["surface"])

    lexicon_messages = []
    for w in words:
        lexicon_messages.append({
            "type": "register_lexicon",
            "predicate": w.lower(), # Fallback if lemma logic is missing
            "language": "en",
            "surface": w
        })

    payloads_str = "\n".join([json.dumps(msg) for msg in lexicon_messages]) + "\n" + payload_str + "\n"

    # Note: Engine's current linearization logic constructs surface outputs by looking up
    # hypothesis predicates (which are usually the node lemmas or surface text) inside its
    # internal linearizer lexicon. For plain English output it expects exactly one hypothesis
    # matching the root of the query graph, meaning the roots surface will be rendered out.

    # Start the rust CLI engine
    print("--- Running Inference Engine (Rust) ---")

    # Ensure binary is built
    subprocess.run(["cargo", "build", "--release", "-p", "csrre"], capture_output=True, check=True)

    cli_process = subprocess.Popen(
        ["./target/release/csrre"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Send the query to the engine
    stdout, stderr = cli_process.communicate(payloads_str)

    if stderr:
        print("STDERR:", stderr)
        pass # Ignore rust logs for clean output unless needed

    if not stdout:
        print("Error: No output from engine")
        return

    try:
        response = json.loads(stdout.strip().split("\n")[-1])
        print("\n--- Output (Plain English) ---")
        print(response.get("surface", "[No output generated]"))
    except json.JSONDecodeError:
        print("Failed to decode engine response:")
        print(stdout)

if __name__ == "__main__":
    main()
