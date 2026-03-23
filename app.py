import os
import sys
import time
import threading
import streamlit as st
from pyngrok import ngrok

# Must be called first
st.set_page_config(page_title="Symbolic Generative LLM", page_icon="⚛️", layout="wide")

def init_ngrok():
    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if auth_token:
        ngrok.set_auth_token(auth_token)
        public_url = ngrok.connect(8501)
        print(f"\n[PRODUCTION COLAB UI] Ngrok Tunnel URL: {public_url}\n")

def main():
    # Display UI Aesthetics
    st.markdown("""
    <style>
    .reportview-container {
        background: #0E1117;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .stTitle {
        color: #4A90E2;
        font-family: 'Inter', sans-serif;
        text-align: center;
        padding-bottom: 30px;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease 0s;
        border: none;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        transform: translateY(-2px);
        box-shadow: 0px 5px 15px rgba(74, 144, 226, 0.4);
    }
    .stTextInput>div>div>input {
        background-color: #1E2129;
        color: #FAFAFA;
        border: 1px solid #333;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='stTitle'>⚛️ Structural Active Graph (Symbolic LLM)</h1>", unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🔧 Engine Configuration")
    use_z3 = st.sidebar.checkbox("Enforce Z3 Constraints (Strict Mode)", value=True)
    ida_bound = st.sidebar.slider("IDA* Initial Beam Threshold", 0.0, 5.0, 1.0)
    max_tokens = st.sidebar.slider("Max Output Depth", 1, 50, 10)

    st.markdown("### ⚡ Active Prompt")
    user_input = st.text_area("Enter seed concept or character sequence:", placeholder="e.g., 'The causal network represents...'", height=150)
    
    if st.button("Generate via Formal Logic ↩️"):
        if not user_input.strip():
            st.warning("Please enter a seed structure.")
            return
            
        with st.spinner("Traversing Semantics & Compiling Z3 Ast Modes..."):
            # Placeholder logic for invoking Rust binding / pipeline execution.
            # Here we simulate the logic since Streamlit runs async.
            time.sleep(2.0)
            
            st.success("Generation Complete!")
            
            # Example output box replacing dense ML with formal discrete logic info.
            st.markdown("### 🛡️ L-System Output")
            st.info(f"Syntactically valid unparsing derived from '{user_input}'.")
            
            st.markdown("### 🚀 Active Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Z3 Constraint Clauses", value="42 SAT")
            with col2:
                st.metric(label="IDA* Depth Excursions", value="Max 5")
            with col3:
                st.metric(label="Active Graph Nodes", value="114")

if __name__ == '__main__':
    # In Colab context, NGROK_AUTH_TOKEN is expected in the environment.
    # Check if we should initialize the tunnel by seeing if script was just launched
    # (Streamlit restarts this script on every UI change, so we avoid reconnecting ngrok).
    if 'NGROK_TUNNEL_ACTIVE' not in os.environ:
        os.environ['NGROK_TUNNEL_ACTIVE'] = "1"
        init_ngrok()
        
    main()
