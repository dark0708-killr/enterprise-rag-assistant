"""Enterprise RAG Assistant - Streamlit Production Web Interface.

Features:
- Hybrid Search (Dense Vector + BM25) with Reciprocal Rank Fusion (RRF)
- Cross-Encoder Re-ranking
- Grounded Source Citations with chunk metadata & confidence scores
- RAG Triad Real-Time Evaluation (Faithfulness, Relevance, Precision)
- Multi-format ingestion (PDF, Markdown, TXT, CSV, JSON)
"""
import io
import time
from pathlib import Path
import streamlit as st

from src.document_loader import Document, DocumentLoader
from src.chunker import RecursiveCharacterChunker
from src.rag_engine import RAGEngine

APP_DIR = Path(__file__).parent
SAMPLE_DOCS_DIR = APP_DIR / "data" / "sample_docs"

st.set_page_config(
    page_title="Enterprise RAG Assistant | Hybrid Search & Citations",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism and modern UI elements
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #111a2e 0%, #080c14 90%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Metrics box */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(8px);
        margin-bottom: 0.75rem;
    }

    /* Citation badge */
    .citation-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 6px;
        padding: 0.15rem 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }

    .badge-verdict {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 700;
    }

    /* Example prompt button */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #38bdf8;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "engine" not in st.session_state:
    with st.spinner("Initializing Hybrid Index & Pre-loading Knowledge Base..."):
        engine = RAGEngine(embedder_provider="local", llm_provider="demo")
        if SAMPLE_DOCS_DIR.exists():
            engine.load_directory(SAMPLE_DOCS_DIR)
        st.session_state.engine = engine
        st.session_state.chat_history = []


engine: RAGEngine = st.session_state.engine

# ==========================================
# SIDEBAR: Architecture & Controls
# ==========================================
with st.sidebar:
    st.markdown("### 🧠 Enterprise RAG Engine")
    st.caption("Dense Vector + BM25 Sparse Hybrid Architecture")

    # Knowledge Base Status
    with st.container():
        st.markdown("#### 📚 Knowledge Base Status")
        col1, col2 = st.columns(2)
        col1.metric("Documents", len(engine.documents))
        col2.metric("Indexed Chunks", len(engine.chunks))

    st.divider()

    # RAG Pipeline Configuration
    st.markdown("#### ⚙️ Retrieval Parameters")
    alpha = st.slider(
        "Hybrid Search Balance (Alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="0.0 = 100% BM25 Sparse Lexical | 1.0 = 100% Dense Semantic Vectors | 0.5 = Balanced RRF"
    )

    top_k = st.slider(
        "Top-K Citations",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of grounded context chunks retrieved"
    )

    enable_reranking = st.toggle(
        "Cross-Encoder Re-ranking",
        value=True,
        help="Two-stage joint relevance re-ranking to prune irrelevant context chunks"
    )

    st.divider()

    # LLM Provider Configuration
    st.markdown("#### 🤖 LLM & Generator Settings")
    llm_choice = st.selectbox(
        "LLM Provider",
        ["Built-in Context Synthesizer (Free Demo)", "OpenAI (GPT-4o-mini)", "Google Gemini (1.5 Flash)"],
        index=0
    )

    api_key_input = None
    if "OpenAI" in llm_choice:
        api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        engine.llm_provider = "openai"
        engine.api_key = api_key_input
    elif "Gemini" in llm_choice:
        api_key_input = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
        engine.llm_provider = "gemini"
        engine.api_key = api_key_input
    else:
        engine.llm_provider = "demo"

    st.divider()

    # Dynamic File Ingestion
    st.markdown("#### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, Markdown, TXT, or CSV",
        type=["pdf", "md", "txt", "csv", "json"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Index Uploaded Files", use_container_width=True):
            new_docs = []
            for up in uploaded_files:
                doc = DocumentLoader.load_bytes(up.name, up.getvalue())
                if doc:
                    new_docs.append(doc)
            if new_docs:
                engine.load_and_index_documents(new_docs)
                st.success(f"Indexed {len(new_docs)} new document(s)!")
                st.rerun()

    # Document Library Preview
    with st.expander("📖 Active Document Library"):
        for doc in engine.documents:
            st.markdown(f"**📄 {doc.metadata.get('filename', 'Doc')}** ({doc.metadata.get('file_type', 'txt')})")
            st.caption(f"{len(doc.content.split())} words | {doc.metadata.get('size_bytes', 0):,} bytes")


# ==========================================
# MAIN INTERFACE: Enterprise Assistant
# ==========================================

st.title("🧠 Enterprise Hybrid RAG Assistant")
st.markdown("""
Production Retrieval-Augmented Generation featuring **Dense Vector Embeddings**, **BM25 Sparse Retrieval**, **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking**, and **RAG Triad Hallucination Guardrails**.
""")

# Sample Prompts Bar
st.markdown("##### 💡 Try an enterprise query:")
prompt_cols = st.columns(3)

p1 = "What was the Q3 2024 revenue, gross profit, and growth rate?"
p2 = "What are the RPO and RTO disaster recovery targets?"
p3 = "What SLA credits apply if uptime drops below 99%?"

selected_prompt = None
if prompt_cols[0].button(f"📊 Q3 Revenue & Growth", use_container_width=True):
    selected_prompt = p1
if prompt_cols[1].button(f"🛡️ Disaster Recovery RPO/RTO", use_container_width=True):
    selected_prompt = p2
if prompt_cols[2].button(f"📜 SLA Uptime Penalties", use_container_width=True):
    selected_prompt = p3

# Display Chat History
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["user"])

    with st.chat_message("assistant"):
        st.markdown(turn["answer"])

        # Collapsible Citations
        with st.expander(f"🔍 Verified Citations ({len(turn['citations'])}) & RAG Triad Metrics"):
            # Metrics Row
            m = turn["metrics"]
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Faithfulness", f"{m.faithfulness * 100:.1f}%")
            m_col2.metric("Answer Relevance", f"{m.answer_relevance * 100:.1f}%")
            m_col3.metric("Context Precision", f"{m.context_precision * 100:.1f}%")
            m_col4.metric("Overall RAG Score", f"{m.overall_score * 100:.1f}%")

            st.markdown(f"**Evaluation Verdict:** <span class='badge-verdict'>{m.verdict}</span>", unsafe_allow_html=True)
            st.caption(f"Pipeline: `{turn['retrieval_method']}`")

            st.divider()
            st.markdown("#### Retrieved Chunks:")
            for cite in turn["citations"]:
                st.markdown(f"**Rank #{cite.rank}** | <span class='citation-tag'>{cite.chunk.doc_id}</span> "
                            f"(Hybrid Score: `{cite.hybrid_score:.4f}` | Dense: `{cite.dense_score:.4f}` | BM25: `{cite.sparse_score:.4f}`)",
                            unsafe_allow_html=True)
                st.info(cite.chunk.text)

# Chat Input Handler
user_input = st.chat_input("Ask any question about enterprise financials, security policies, SLAs, or uploaded files...")
query_to_run = selected_prompt or user_input

if query_to_run:
    # Render user query
    with st.chat_message("user"):
        st.markdown(query_to_run)

    # Process RAG response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving hybrid chunks & synthesizing answer..."):
            start_t = time.time()
            response = engine.query(
                user_query=query_to_run,
                top_k=top_k,
                alpha=alpha,
                enable_reranking=enable_reranking
            )
            elapsed = time.time() - start_t

        st.markdown(response.answer)

        # Citations & Metrics Accordion
        with st.expander(f"🔍 Verified Citations ({len(response.citations)}) & RAG Triad Metrics ({elapsed:.2f}s)"):
            m = response.metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Faithfulness", f"{m.faithfulness * 100:.1f}%")
            m_col2.metric("Answer Relevance", f"{m.answer_relevance * 100:.1f}%")
            m_col3.metric("Context Precision", f"{m.context_precision * 100:.1f}%")
            m_col4.metric("Overall RAG Score", f"{m.overall_score * 100:.1f}%")

            st.markdown(f"**Evaluation Verdict:** <span class='badge-verdict'>{m.verdict}</span>", unsafe_allow_html=True)
            st.caption(f"Pipeline: `{response.retrieval_method}` | Ingestion: {response.chunks_analyzed} chunks evaluated")

            st.divider()
            st.markdown("#### Grounded Source Chunks:")
            for cite in response.citations:
                st.markdown(f"**Rank #{cite.rank}** | <span class='citation-tag'>{cite.chunk.doc_id}</span> "
                            f"(Hybrid Score: `{cite.hybrid_score:.4f}` | Dense: `{cite.dense_score:.4f}` | BM25: `{cite.sparse_score:.4f}`)",
                            unsafe_allow_html=True)
                st.info(cite.chunk.text)

    # Store in history
    st.session_state.chat_history.append({
        "user": query_to_run,
        "answer": response.answer,
        "citations": response.citations,
        "metrics": response.metrics,
        "retrieval_method": response.retrieval_method
    })
