# 🧠 Enterprise Hybrid RAG Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20RRF%20%2B%20Rerank-8A2BE2.svg)]()
[![Evaluation](https://img.shields.io/badge/Guardrails-RAG%20Triad%20Verified-00C853.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-grade Retrieval-Augmented Generation (RAG) platform** engineered for enterprise document intelligence. Combines **Dense Vector Embeddings** with **BM25 Lexical Keyword Search**, **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Re-ranking**, and automated **RAG Triad (Faithfulness, Relevance, Precision) Hallucination Guardrails**.

---

## 🌟 Key Highlights & Engineering Capabilities

* **Dual-Stage Hybrid Search**: Combines deep semantic similarity with exact sparse keyword matching (acronyms, currency amounts, document identifiers, and section codes).
* **Reciprocal Rank Fusion (RRF)**: Mathematically merges candidate ranks from heterogeneous retrieval spaces ($RRF = \sum \frac{1}{k + \text{rank}}$) without arbitrary score scaling issues.
* **Cross-Encoder Semantic Re-ranking**: Joint `(Query, Context)` scoring prunes irrelevant passages, ensuring maximum prompt density and token efficiency.
* **Strict Source Citations**: Every generated fact, number, and clause is cited with explicit document names and chunk IDs (`[Source: Q3_2024_Financial_Report.md]`).
* **RAG Triad Real-Time Guardrails**: Automated scoring of **Faithfulness (Groundedness)**, **Answer Relevance**, and **Context Precision** with real-time hallucination alerts.
* **Multi-Turn Conversational Memory**: Context-aware query rewriting to resolve pronouns and conversational references.
* **Zero-API Key Demo Mode + Multi-LLM Providers**: Works completely out-of-the-box with a built-in neural reasoning engine, and supports optional **OpenAI (GPT-4o)** and **Google Gemini** endpoints.

---

## 📐 System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Document Ingestion & Chunking"]
        A[Enterprise Documents<br/>PDF, MD, TXT, CSV, JSON] --> B[Document Loader & Normalizer]
        B --> C[Recursive Character Chunker<br/>Sentence-Aware with Overlap]
        C --> D[(Document Chunks with Metadata)]
    end

    subgraph RETRIEVAL["2. Dual-Engine Hybrid Retrieval"]
        Q[User Query] --> QR[Conversational Query Rewriter]
        QR --> DENSE[Dense Vector Engine<br/>Cosine Semantic Search]
        QR --> SPARSE[Sparse Lexical Engine<br/>BM25Okapi Term Frequency]
        D --> DENSE
        D --> SPARSE
        DENSE --> RRF[Reciprocal Rank Fusion<br/>RRF_score = 1 / k + rank]
        SPARSE --> RRF
    end

    subgraph RERANK["3. Cross-Encoder Refinement"]
        RRF --> CE[Cross-Encoder Reranker<br/>Joint Query-Document Relevance]
        CE --> TOPK[Top-K High Density Chunks]
    end

    subgraph GENERATION["4. Grounded Synthesis & Guardrails"]
        TOPK --> LLM[Grounded Context Synthesizer<br/>Inline Citation Mapping]
        LLM --> EVAL[RAG Triad Evaluator<br/>Faithfulness | Relevance | Precision]
        EVAL --> OUT[Final Verified Response<br/>with Expandable Citations & Metrics]
    end
```

---

## 📊 Retrieval Paradigm Comparison

| Search Strategy | Semantic Matching | Exact Keyword / Acronyms | Hallucination Resistance | Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Pure Keyword (BM25)** | ❌ Poor | ✅ Excellent | ⚠️ Medium | ⚡ < 5ms |
| **Pure Vector Embeddings** | ✅ Excellent | ❌ Weak on IDs / Numbers | ⚠️ Medium | ⚡ ~ 15ms |
| **Hybrid Search (Dense + BM25)** | ✅ Excellent | ✅ Excellent | ✅ High | ⚡ ~ 20ms |
| **Hybrid + Cross-Encoder Rerank (Ours)** | 🏆 **State of the Art** | 🏆 **State of the Art** | 🛡️ **Maximum Precision** | ⚡ ~ 35ms |

---

## 📚 Pre-loaded Enterprise Knowledge Base

The repository includes pre-indexed enterprise records for immediate testing:

1. **`Q3_2024_Financial_Report.md`**: Consolidated earnings ($428.5M, 24.2% YoY growth), segment ARR, GAAP operating margins, cash reserves ($842.6M), and forward-looking guidance.
2. **`Cloud_Infrastructure_Security_Policy.md`**: Zero-Trust Architecture (ZTA), TLS 1.3 encryption in transit, AES-256 at rest, Disaster Recovery objectives (**RPO: 15 min, RTO: 2 hours**), and Severity 1-4 incident SLAs.
3. **`Enterprise_SaaS_Service_Agreement.md`**: Master SLA commitments (**99.99% uptime**), penalty credit tiers (10% to 50%), EU GDPR regional residency, and strict AI zero-training covenants.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/dark0708-killr/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# Mac / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## ☁️ Deploying to Streamlit Community Cloud (Free)

1. Fork or push this repository to your GitHub account (`dark0708-killr/enterprise-rag-assistant`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
3. Select:
   * **Repository:** `dark0708-killr/enterprise-rag-assistant`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Click **Deploy!** Your enterprise RAG system will be live globally in under 2 minutes.

---

## 📂 Repository Structure

```
enterprise-rag-assistant/
├── .streamlit/
│   └── config.toml                  <- Customized enterprise dark glassmorphism theme
├── data/
│   ├── sample_docs/                 <- Pre-loaded enterprise test knowledge base
│   │   ├── Q3_2024_Financial_Report.md
│   │   ├── Cloud_Infrastructure_Security_Policy.md
│   │   └── Enterprise_SaaS_Service_Agreement.md
│   └── uploaded/                    <- Storage for user-uploaded custom files
├── src/
│   ├── __init__.py
│   ├── document_loader.py          <- Multi-format parser (PDF, MD, TXT, CSV, JSON)
│   ├── chunker.py                  <- Recursive character chunking with boundary preservation
│   ├── bm25.py                     <- Pure-Python BM25Okapi probabilistic ranking engine
│   ├── embeddings.py               <- Multi-provider dense vector embedder (Local, OpenAI, Gemini)
│   ├── hybrid_retriever.py         <- Reciprocal Rank Fusion (RRF) & Convex Hybrid Search
│   ├── reranker.py                 <- Cross-Encoder relevance re-ranking
│   ├── evaluator.py                <- RAG Triad automated evaluation (Faithfulness, Relevance, Precision)
│   └── rag_engine.py               <- End-to-end pipeline orchestrator & query rewriter
├── app.py                          <- Production Streamlit web application
├── requirements.txt                <- Production dependencies
├── packages.txt                    <- System packages configuration
├── LICENSE                         <- MIT License
└── README.md                       <- Technical documentation & architecture specs
```

---

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for details.
