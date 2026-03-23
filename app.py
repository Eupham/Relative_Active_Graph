import streamlit as st
import sys, json, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lcs" / "induction"))

from tokenizer import tokenize
from text_to_graph import sentence_to_mtlg
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Relative Active Graph (RAG)", layout="wide")
st.title("Relative Active Graph (RAG)")


def _binary() -> Path:
    p = Path("target/release/csrre")
    return p if p.exists() else Path("target/debug/csrre")


tab_train, tab_infer = st.tabs(["Train", "Inference & Visualization"])

# ─── TRAIN ────────────────────────────────────────────────────────────────────

with tab_train:
    st.header("Train on C4 (Teacher Forcing)")
    st.caption("No bootstrap phase. The engine learns continuously from the data stream.")

    col1, col2, col3, col4 = st.columns(4)
    with col1: language        = st.text_input("Language", value="en")
    with col2: max_sentences   = st.number_input("Max sentences", 50, 50_000, 500, 50)
    with col3: epochs          = st.number_input("Epochs", 1, 20, 1)
    with col4: category_warmup = st.number_input("Category warmup (passages)", 10, 1000, 100, 10)

    if st.button("Start Training"):
        binary = _binary()
        if not binary.exists():
            st.error(f"Binary not found at {binary}. Run `cargo build --release` first.")
        else:
            sys.path.insert(0, str(Path(__file__).parent / "lcs" / "training"))
            from sequential_trainer import SequentialTrainer, TrainingConfig

            config = TrainingConfig(
                language=language, epochs=int(epochs),
                max_sentences=int(max_sentences), binary=binary,
                category_warmup=int(category_warmup),
            )
            with st.spinner("Teacher-forcing in progress…"):
                stats_list = SequentialTrainer(config).train()

            if stats_list:
                f = stats_list[-1]
                st.success(f"Complete — {f.passages} passages · {f.sentences} sentences · "
                           f"mean quality = {f.mean_quality:.4f}")
                st.session_state["language"] = language
                st.session_state["trained"]  = True
            else:
                st.error("No output. Check binary path and mC4 connectivity.")

# ─── INFERENCE ────────────────────────────────────────────────────────────────

with tab_infer:
    st.header("Inference & Visualization")

    if not st.session_state.get("trained", False):
        st.info("Run the Train tab first, or train externally with "
                "`python -m lcs.training.sequential_trainer`.")

    language_infer = st.text_input("Language", value=st.session_state.get("language", "en"), key="il")
    test_sentence  = st.text_input("Input sentence",
                                   value="The scientist discovered a new particle.")

    if st.button("Analyse & Generate"):
        binary = _binary()

        with st.spinner("Tokenising…"):
            try:
                s = tokenize(test_sentence, language_infer)
                g = sentence_to_mtlg(s)
            except Exception as exc:
                st.error(f"Tokenisation failed: {exc}")
                st.stop()

        st.subheader("Parse")
        st.dataframe([{"Token": n.text, "Lemma": n.lemma, "Mode": n.modal_mode, "Arity": n.arity}
                      for n in g.nodes], use_container_width=True)

        if g.nodes and g.edges:
            with st.expander("MTLG Graph", expanded=False):
                G_nx = nx.DiGraph()
                for n in g.nodes: G_nx.add_node(n.token_id, label=n.text)
                for e in g.edges: G_nx.add_edge(e.src_id, e.dst_id)
                fig, ax = plt.subplots(figsize=(max(6, len(g.nodes)), 3))
                pos = nx.spring_layout(G_nx, seed=42)
                nx.draw(G_nx, pos, ax=ax,
                        labels={n.token_id: n.text for n in g.nodes},
                        node_color="#4C72B0", node_size=700, font_size=9,
                        font_color="white", edge_color="#888", arrows=True)
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("Generated Output")
        if not binary.exists():
            st.warning(f"Binary not found at {binary}.")
        else:
            wire_nodes = [{"id": n.token_id, "surface": n.text, "score": 0.9,
                            "mode": n.modal_mode, "cat": n.category_id, "arity": n.arity}
                          for n in g.nodes]
            wire_edges = [{"id": i + 1, "src": e.src_id, "dst": e.dst_id,
                            "mode": e.modal_mode, "weight": 1.0}
                          for i, e in enumerate(g.edges)]

            lex_msgs  = [json.dumps({"type": "register_lexicon",
                                      "predicate": n["surface"].lower(),
                                      "language": language_infer,
                                      "surface": n["surface"]}) for n in wire_nodes]
            query_msg = json.dumps({"type": "query", "text": test_sentence,
                                     "situation_id": 1, "language": language_infer,
                                     "nodes": wire_nodes, "edges": wire_edges})
            payload   = "\n".join(lex_msgs + [query_msg]) + "\n"

            with st.spinner("Running inference…"):
                try:
                    proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    stdout, stderr = proc.communicate(payload, timeout=30)
                except subprocess.TimeoutExpired:
                    st.error("Engine timed out.")
                    st.stop()
                except Exception as exc:
                    st.error(f"Engine failed: {exc}")
                    st.stop()

            if stdout.strip():
                lines = [l for l in stdout.strip().split("\n") if l.strip()]
                try:
                    r = json.loads(lines[-1])
                    surface = r.get("surface", "")
                    if surface and "[no output" not in surface:
                        st.success(surface)
                    else:
                        st.warning("Engine returned no surface output.")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Satisfied",  str(r.get("satisfied", "—")))
                    c2.metric("Quality",    f"{r.get('quality', 0):.3f}")
                    c3.metric("Depth used", str(r.get("depth", "—")))
                except json.JSONDecodeError:
                    st.error(f"Could not parse engine response:\n```\n{stdout[:400]}\n```")
            else:
                st.error(f"No output.\n\nstderr:\n```\n{(stderr or '')[:300]}\n```")
