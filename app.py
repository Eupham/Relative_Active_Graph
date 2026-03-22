import streamlit as st
import sys
from pathlib import Path

# Ensure lcs directory is in the path
sys.path.insert(0, str(Path(__file__).parent / "lcs" / "induction"))
sys.path.insert(0, str(Path(__file__).parent / "lcs" / "training"))

from mc4_stream import stream_mc4, _split_sentences
from c4_sequence_extractor import _sentence_to_sequence
from mtlg_inducer import MtlgInducer
from trd_bootstrap import TrdBootstrapper

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
        train_samples = st.number_input("Max Samples", min_value=10, max_value=5000, value=200, step=50)
    with col3:
        n_clusters = st.number_input("TRD Clusters", min_value=2, max_value=100, value=16, step=2)

    if st.button("Start Training"):
        with st.spinner("Streaming mC4 and collecting sentences..."):
            from mc4_stream import stream_mc4, _split_sentences
            from c4_sequence_extractor import _sentence_to_sequence
            sentences = []
            for item in stream_mc4(language, max_samples=train_samples):
                for s in _split_sentences(item.get("text", "")):
                    sentences.append(s)
            st.success(f"Collected {len(sentences)} sentences.")

        with st.spinner("Inducing character lexicon..."):
            inducer = MtlgInducer(language)
            for sentence in sentences:
                seq = _sentence_to_sequence(sentence, trd_id=0)
                for step in seq.steps:
                    inducer.lexicon.update(step.lemma, "diamond", 0, 0, 1.0)
            inducer.lexicon.normalize()
            lexicon = inducer.lexicon
            st.success(f"Lexicon: {len(lexicon.entries)} characters.")

        with st.spinner("Bootstrapping TRDs..."):
            bootstrapper = TrdBootstrapper(n_clusters=n_clusters)
            trds = bootstrapper.bootstrap(sentences, language)
            st.success(f"Bootstrapped {len(trds)} TRDs.")

        st.session_state["lexicon"]      = lexicon
        st.session_state["bootstrapper"] = bootstrapper
        st.session_state["language"]     = language

with tab2:
    st.header("Inference & Visualization")

    if "lexicon" not in st.session_state:
        st.warning("Please run the training pipeline first on the 'Train' tab.")
    else:
        test_sentence = st.text_input("Test Sentence", "The scientist discovered a new particle in the laboratory.")

        if st.button("Analyze Sentence"):
            lexicon = st.session_state["lexicon"]
            bootstrapper = st.session_state["bootstrapper"]
            language = st.session_state["language"]

            with st.spinner("Analyzing..."):
                seq = _sentence_to_sequence(test_sentence, trd_id=0)
                node_data = [{"Char": repr(s.text), "node_id": f"{s.node_id:#014x}"}
                             for s in seq.steps]
                st.dataframe(node_data)
