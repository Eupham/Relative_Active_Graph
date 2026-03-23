"""End-to-end English test. No UD, no Stanza."""
import sys, json, subprocess
from pathlib import Path

sys.path.insert(0, str(Path("lcs/induction").absolute()))

from tokenizer import tokenize
from text_to_graph import sentence_to_mtlg


def build_cli_query(sentence: str, situation_id: int) -> dict:
    language = "en"
    s    = tokenize(sentence, language)
    g    = sentence_to_mtlg(s.tokens, language)
    nodes = [{"id": n.token_id, "surface": n.text, "score": 0.9,
               "mode": n.modal_mode, "cat": n.category_id, "arity": n.arity}
             for n in g.nodes]
    edges = [{"id": i + 1, "src": e.src_id, "dst": e.dst_id, "mode": e.modal_mode, "weight": 1.0}
             for i, e in enumerate(g.edges)]
    return {"type": "query", "text": sentence, "situation_id": situation_id,
            "language": language, "nodes": nodes, "edges": edges}


def main():
    sentence = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Alice discovers a particle."
    print(f"--- Input ---\n{sentence}\n")

    query   = build_cli_query(sentence, 42)
    lex_msgs = [json.dumps({"type": "register_lexicon", "predicate": n["surface"].lower(),
                             "language": "en", "surface": n["surface"]})
                for n in query["nodes"]]
    payload  = "\n".join(lex_msgs + [json.dumps(query)]) + "\n"

    binary = Path("target/release/csrre")
    if not binary.exists():
        binary = Path("target/debug/csrre")
    if not binary.exists():
        print("Binary not found. Run `cargo build --release`.")
        return

    proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, _ = proc.communicate(payload)

    if stdout.strip():
        lines = [l for l in stdout.strip().split("\n") if l.strip()]
        try:
            response = json.loads(lines[-1])
            print("--- Output ---")
            print(response.get("surface", "[no output]"))
        except json.JSONDecodeError:
            print(f"Parse error:\n{stdout}")
    else:
        print("No output from engine.")


if __name__ == "__main__":
    main()
