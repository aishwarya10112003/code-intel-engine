"""
app.py — Phase 8. A simple web chat UI for the Code-Intel Engine (Streamlit).

Run it with:
    ./.venv/bin/streamlit run app.py

It opens in your browser. Ask questions about whatever codebase/docs you indexed
(build_index.py), pick the retrieval strategy in the sidebar, and optionally turn on the
agentic loop for hard multi-part questions.
"""
from __future__ import annotations

import os

import streamlit as st

from src.agent import AgenticRAG
from src.rag import build_pipeline

# On Streamlit Cloud the API key comes from the platform's Secrets box (st.secrets).
# Copy it into the environment so the existing Groq client (which reads os.environ) finds it.
# Locally there's no secrets file, so this is skipped and the .env file handles the key.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Code-Intel Engine", page_icon="🧠", layout="wide")


# @st.cache_resource loads the heavy models ONCE and reuses them across questions,
# instead of reloading the embedding/rerank models on every click.
@st.cache_resource(show_spinner="Loading models & index...")
def get_pipeline(config: str):
    return build_pipeline(config=config, chunks_path="chunks.json")


st.title("🧠 Code-Intel Engine")
st.caption("Ask questions about your indexed codebase or docs — answers come with citations.")

with st.sidebar:
    st.header("Settings")
    config = st.selectbox(
        "Retrieval strategy",
        ["hybrid_rerank", "hybrid", "dense"],
        help="dense = meaning only · hybrid = + keywords · hybrid_rerank = + a sharp reranker",
    )
    agent_mode = st.toggle(
        "Agentic mode",
        value=False,
        help="Decomposes multi-part questions and self-checks the answer (slower, thorough).",
    )
    st.markdown("---")
    st.caption("Index your data first:\n\n`python ingest.py <folder>`\n\n`python build_index.py chunks.json`")

question = st.text_input("Your question", placeholder="e.g. how does placing an order prevent overselling?")

if question:
    try:
        pipeline = get_pipeline(config)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Thinking..."):
        if agent_mode:
            answer, hits, trace = AgenticRAG(pipeline, max_retries=1).answer(question)
        else:
            answer, hits = pipeline.answer(question)
            trace = None

    st.subheader("Answer")
    st.write(answer)

    if trace:
        st.info(
            f"🧩 Sub-questions: {trace['sub_questions']}\n\n"
            f"✅ Self-check faithfulness: {trace['faithfulness_scores']}  "
            f"(retries: {trace['retries']})"
        )

    st.subheader("Sources")
    for i, hit in enumerate(hits, 1):
        with st.expander(f"[{i}] {hit['chunk_id']}"):
            st.code(hit["content"])
