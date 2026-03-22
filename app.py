import streamlit as st
import sys
from pathlib import Path

# Ensure lcs directory is in the path
sys.path.insert(0, str(Path(__file__).parent / "lcs" / "induction"))

from mc4_stream import stream_mc4
from ud_parser import UdParser
from ud_to_mtlg import ud_tree_to_mtlg
from morphological_fst import preprocess_for_type_assignment
from mtlg_inducer import MtlgInducer
from trd_bootstrap import TrdBootstrapper, graph_to_modal_vector

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

st.set_page_config(page_title="Relative Active Graph (RAG) Training & Inference", layout="wide")

st.title("Relative Active Graph (RAG) Training & Inference")

tab1, tab2 = st.tabs(["Train", "Inference & Visualization"])

with tab1:
    st.header("Train on C4")

    col1, col2, col3 = st.columns(3)
    with col1:
        language = st.text_input("Language Code (e.g. en, fr, de)", value="en")
    with col2:
        max_samples = st.number_input("Max Samples", min_value=10, max_value=5000, value=200, step=50)
    with col3:
        n_clusters = st.number_input("TRD Clusters", min_value=2, max_value=100, value=16, step=2)

    if st.button("Start Training"):
        st.write(f"Streaming {max_samples} sentences from mC4 ({language})...")

        parser = UdParser(language)
        graphs = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        count = 0
        for item in stream_mc4(language, max_samples=max_samples):
            try:
                tokens = preprocess_for_type_assignment(item["text"], language)
                sentence = " ".join(t.split("[")[0] for t in tokens)
                tree = parser.parse(sentence)
                graph = ud_tree_to_mtlg(tree)
                if graph.nodes:
                    graphs.append(graph)
            except Exception as e:
                st.warning(f"Error parsing sentence: {e}")
                pass

            count += 1
            progress_bar.progress(count / max_samples)
            status_text.text(f"Processed {count}/{max_samples} sentences...")

        st.success(f"Successfully created {len(graphs)} MTLG graphs.")

        with st.spinner("Inducing MTLG lexicon..."):
            inducer = MtlgInducer(language)
            lexicon = inducer.induce_from_stream(iter(graphs), max_trees=len(graphs))
            st.success(f"Lexicon created with {len(lexicon.entries)} lemmas.")

        with st.spinner("Bootstrapping TRDs..."):
            bootstrapper = TrdBootstrapper(n_clusters=n_clusters)
            trds = bootstrapper.bootstrap(graphs, language)
            st.success(f"Bootstrapped {len(trds)} TRDs.")

        st.session_state["parser"] = parser
        st.session_state["lexicon"] = lexicon
        st.session_state["bootstrapper"] = bootstrapper
        st.session_state["language"] = language

        st.balloons()

with tab2:
    st.header("Inference & Visualization")

    if "parser" not in st.session_state:
        st.warning("Please run the training pipeline first on the 'Train' tab.")
    else:
        test_sentence = st.text_input("Test Sentence", "The scientist discovered a new particle in the laboratory.")

        if st.button("Analyze Sentence"):
            parser = st.session_state["parser"]
            lexicon = st.session_state["lexicon"]
            bootstrapper = st.session_state["bootstrapper"]
            language = st.session_state["language"]

            with st.spinner("Analyzing..."):
                tokens = preprocess_for_type_assignment(test_sentence, language)
                clean = " ".join(t.split("[")[0] for t in tokens)
                tree = parser.parse(clean)
                g = ud_tree_to_mtlg(tree)

                # TRD Assignment
                trd_label = "unknown"
                trd_patterns = []
                if bootstrapper._model and bootstrapper.vocab:
                    vec = graph_to_modal_vector(g, bootstrapper.vocab)
                    trd_id = bootstrapper._model.predict(vec)
                    for t in bootstrapper.trds:
                        if t.trd_id == trd_id:
                            trd_label = t.label
                            trd_patterns = t.infon_patterns
                            break

                st.subheader("Analysis Results")
                st.write(f"**Assigned TRD Cluster:** {trd_label}")
                if trd_patterns:
                    st.write(f"**TRD Keys:** {', '.join(trd_patterns)}")

                st.write("**Node Lexicon Lookup:**")
                node_data = []
                for node in g.nodes:
                    entry = lexicon.best_type(node.lemma)
                    node_data.append({
                        "Token": node.text,
                        "UPOS": node.upos,
                        "UCCA Cat": node.category_id,
                        "Lex Mode": entry.modal_mode if entry else "—",
                        "Arity": node.arity,
                        "Obs Count": int(entry.count) if entry else 0
                    })
                st.table(node_data)

                st.subheader("MTLG Dependency Graph")

                G = nx.DiGraph()
                for node in g.nodes:
                    G.add_node(node.token_id, label=node.text, ucca=node.category_id)
                for edge in g.edges:
                    G.add_edge(edge.src_id, edge.dst_id, mode=edge.modal_mode, deprel=edge.deprel)

                UCCA_COLORS = {
                    1: "#DD8452", 6: "#4C72B0", 0: "#55A868",
                    5: "#C44E52", 4: "#8172B3",
                    2: "#937860", 3: "#DA8BC3",
                }
                UCCA_NAMES = {
                    1: "Process", 6: "Participant", 0: "Scene",
                    5: "State", 4: "Adverbial",
                    2: "Connector", 3: "Ground",
                }
                MODE_STYLE = {"diamond": "solid", "box": "dashed", "lozenge": "dotted"}
                MODE_COLOR = {"diamond": "#4C72B0", "box": "#DD8452", "lozenge": "#55A868"}
                MODE_SYM = {"diamond": "◇", "box": "□", "lozenge": "◊"}

                node_colors = [UCCA_COLORS.get(G.nodes[n].get("ucca", 0), "#aaaaaa") for n in G.nodes]
                pos = nx.spring_layout(G, seed=42, k=2.2)

                fig, ax = plt.subplots(figsize=(14, 6))
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, ax=ax, alpha=0.95)
                nx.draw_networkx_labels(G, pos,
                    labels={n: G.nodes[n].get("label", str(n)) for n in G.nodes},
                    font_size=8, font_color="white", font_weight="bold", ax=ax)

                for (u, v, data) in G.edges(data=True):
                    mode = data.get("mode", "diamond")
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                        style=MODE_STYLE.get(mode, "solid"),
                        edge_color=MODE_COLOR.get(mode, "#888"),
                        arrows=True, arrowsize=20, width=2, ax=ax,
                        connectionstyle="arc3,rad=0.1")

                edge_labels = {(e.src_id, e.dst_id): e.deprel for e in g.edges}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

                legend_patches = [mpatches.Patch(color=c, label=UCCA_NAMES[k]) for k, c in UCCA_COLORS.items()]
                mode_lines = [
                    plt.Line2D([0],[0], color=MODE_COLOR[m], lw=2,
                               linestyle=MODE_STYLE[m], label=f"{MODE_SYM[m]} {m}")
                    for m in ("diamond","box","lozenge")
                ]
                ax.legend(handles=legend_patches + mode_lines, loc="lower left", fontsize=8,
                          title="UCCA / Mode", ncol=2)
                ax.set_title(f'MTLG Graph: "{test_sentence}"', fontsize=11, fontweight="bold")
                ax.axis("off")

                st.pyplot(fig)
