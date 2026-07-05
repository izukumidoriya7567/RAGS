"""
backend_ai.py
=============
Query-time backend for the hierarchical Hybrid-Search RAG system.

Flow (LangGraph):

                 ┌────────────────┐
   user query ─► │  classify_node │  (SLM on Groq: "simple" | "complex")
                 └───────┬────────┘
              simple ────┴──── complex
                │                │
     ┌──────────▼─────┐  ┌──────▼───────────┐
     │ rewrite_node   │  │ decompose_node   │  (SLM → sub-queries)
     │ (SLM expands / │  └──────┬───────────┘
     │  rewrites)     │         │
     └──────┬─────────┘  ┌──────▼───────────┐
            │            │ parallel_retrieve│  (ThreadPoolExecutor,
     ┌──────▼─────────┐  │  + rerank each)  │   hybrid search per sub-query)
     │ retrieve_node  │  └──────┬───────────┘
     │ (hybrid+rerank)│  ┌──────▼───────────┐
     └──────┬─────────┘  │ summarize_node   │  (SLM summarizes chunks)
            │            └──────┬───────────┘
            └───────┬───────────┘
             ┌──────▼───────┐
             │ generate_node│  (LLM on Groq → final answer)
             └──────────────┘

- Hybrid search : Qdrant Query API with dense + sparse prefetch fused via RRF
- Reranking     : sentence-transformers CrossEncoder
- SLM           : llama-3.1-8b-instant   (Groq)
- LLM           : llama-3.3-70b-versatile (Groq)

Run:
    export GROQ_API_KEY=...
    export QDRANT_URL=...
    export QDRANT_API_KEY=...
    python backend_ai.py                # interactive CLI
    uvicorn backend_ai:app --reload     # FastAPI server (POST /query)
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from operator import add
from typing import Annotated, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration (must match ingest_ai.py)
# --------------------------------------------------------------------------- #
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "hierarchical_rag")

DENSE_MODEL_NAME = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
CROSS_ENCODER_NAME = os.getenv("CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

SLM_MODEL = os.getenv("GROQ_SLM", "llama-3.1-8b-instant")
LLM_MODEL = os.getenv("GROQ_LLM", "llama-3.3-70b-versatile")

TOP_K_PREFETCH = 20      # candidates fetched per (dense, sparse) leg
TOP_K_FUSED = 12         # after RRF fusion
TOP_K_RERANKED = 5       # after cross-encoder rerank
MAX_PARALLEL_WORKERS = 4

# --------------------------------------------------------------------------- #
# Singletons (loaded once at startup)
# --------------------------------------------------------------------------- #
dense_model = SentenceTransformer(DENSE_MODEL_NAME)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

slm = ChatGroq(model=SLM_MODEL, temperature=0.0)
llm = ChatGroq(model=LLM_MODEL, temperature=0.2)


# --------------------------------------------------------------------------- #
# Hybrid retrieval + cross-encoder rerank
# --------------------------------------------------------------------------- #
def hybrid_search(query: str, top_k: int = TOP_K_FUSED) -> List[dict]:
    """Dense + sparse (BM25) prefetch, fused with Reciprocal Rank Fusion."""
    dense_vec = dense_model.encode(query, normalize_embeddings=True).tolist()
    sparse_emb = next(iter(sparse_model.embed([query])))
    sparse_vec = models.SparseVector(
        indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist()
    )

    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec, using=DENSE_VECTOR_NAME, limit=TOP_K_PREFETCH),
            models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR_NAME, limit=TOP_K_PREFETCH),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text": p.payload["text"],
            "parent_text": p.payload.get("parent_text", p.payload["text"]),
            "source": p.payload.get("source", "unknown"),
            "score": p.score,
        }
        for p in result.points
    ]


def rerank(query: str, chunks: List[dict], top_k: int = TOP_K_RERANKED) -> List[dict]:
    """Cross-encoder rerank of retrieved child chunks."""
    if not chunks:
        return []
    scores = cross_encoder.predict([(query, c["text"]) for c in chunks])
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

    # Deduplicate by parent so we don't feed the same parent context twice
    seen, unique = set(), []
    for c in ranked:
        key = c["parent_text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def retrieve_and_rerank(query: str) -> List[dict]:
    return rerank(query, hybrid_search(query))


# --------------------------------------------------------------------------- #
# LangGraph state
# --------------------------------------------------------------------------- #
class RAGState(TypedDict):
    original_query: str
    query_type: Optional[Literal["simple", "complex"]]
    rewritten_query: Optional[str]
    sub_queries: List[str]
    retrieved_chunks: Annotated[List[dict], add]
    summaries: List[str]
    answer: Optional[str]


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #
def classify_node(state: RAGState) -> dict:
    """SLM classifies the query as 'simple' or 'complex'."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You classify user queries for a RAG system.\n"
                "A query is SIMPLE if it asks one focused thing answerable with a "
                "single retrieval (a fact, definition, single how-to).\n"
                "A query is COMPLEX if it is multi-part, comparative, requires "
                "reasoning across multiple topics, or contains multiple questions.\n"
                'Respond with ONLY one word: "simple" or "complex".',
            ),
            ("human", "{query}"),
        ]
    )
    raw = (prompt | slm).invoke({"query": state["original_query"]}).content.strip().lower()
    query_type = "complex" if "complex" in raw else "simple"
    print(f"[classify] -> {query_type}")
    return {"query_type": query_type}


def route_by_type(state: RAGState) -> str:
    return state["query_type"]


def rewrite_node(state: RAGState) -> dict:
    """SIMPLE path: SLM expands/rewrites the query for better retrieval."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user's search query to maximize retrieval quality. "
                "Expand abbreviations, add key synonyms, make implicit intent explicit. "
                "Keep it a single query. Output ONLY the rewritten query, nothing else.",
            ),
            ("human", "{query}"),
        ]
    )
    rewritten = (prompt | slm).invoke({"query": state["original_query"]}).content.strip()
    print(f"[rewrite] {rewritten}")
    return {"rewritten_query": rewritten}


def retrieve_node(state: RAGState) -> dict:
    """SIMPLE path: hybrid retrieval + rerank using the rewritten query."""
    query = state["rewritten_query"] or state["original_query"]
    chunks = retrieve_and_rerank(query)
    print(f"[retrieve] {len(chunks)} chunks")
    return {"retrieved_chunks": chunks}


def decompose_node(state: RAGState) -> dict:
    """COMPLEX path: SLM breaks the query into sub-queries."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Decompose the user's complex question into 2-5 self-contained "
                "sub-queries, each retrievable on its own. "
                'Return ONLY a JSON array of strings, e.g. ["q1", "q2"]. No prose.',
            ),
            ("human", "{query}"),
        ]
    )
    raw = (prompt | slm).invoke({"query": state["original_query"]}).content.strip()
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        sub_queries = json.loads(match.group(0)) if match else [state["original_query"]]
        sub_queries = [q for q in sub_queries if isinstance(q, str) and q.strip()][:5]
    except (json.JSONDecodeError, AttributeError):
        sub_queries = [state["original_query"]]
    if not sub_queries:
        sub_queries = [state["original_query"]]
    print(f"[decompose] {sub_queries}")
    return {"sub_queries": sub_queries}


def parallel_retrieve_node(state: RAGState) -> dict:
    """COMPLEX path: retrieve for all sub-queries in parallel."""
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as pool:
        results = list(pool.map(retrieve_and_rerank, state["sub_queries"]))

    all_chunks, seen = [], set()
    for sub_q, chunks in zip(state["sub_queries"], results):
        for c in chunks:
            key = c["text"][:200]
            if key not in seen:
                seen.add(key)
                c["sub_query"] = sub_q
                all_chunks.append(c)
    print(f"[parallel_retrieve] {len(all_chunks)} unique chunks")
    return {"retrieved_chunks": all_chunks}


def summarize_node(state: RAGState) -> dict:
    """COMPLEX path: SLM summarizes retrieved chunks (grouped by sub-query)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize the following retrieved passages into a dense, factual "
                "summary that preserves all details relevant to the question. "
                "No fluff, no preamble.",
            ),
            ("human", "Question: {question}\n\nPassages:\n{passages}"),
        ]
    )
    chain = prompt | slm

    groups: dict[str, List[dict]] = {}
    for c in state["retrieved_chunks"]:
        groups.setdefault(c.get("sub_query", "general"), []).append(c)

    def summarize_group(item):
        sub_q, chunks = item
        passages = "\n\n---\n\n".join(c["parent_text"] for c in chunks)
        summary = chain.invoke({"question": sub_q, "passages": passages}).content.strip()
        return f"### Sub-question: {sub_q}\n{summary}"

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as pool:
        summaries = list(pool.map(summarize_group, groups.items()))

    print(f"[summarize] {len(summaries)} summaries")
    return {"summaries": summaries}


def generate_node(state: RAGState) -> dict:
    """Final answer with the LLM."""
    if state["query_type"] == "complex":
        context = "\n\n".join(state["summaries"])
    else:
        context = "\n\n---\n\n".join(c["parent_text"] for c in state["retrieved_chunks"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise assistant. Answer the user's question using ONLY "
                "the provided context. If the context is insufficient, say so. "
                "Cite sources inline when useful.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    answer = (prompt | llm).invoke(
        {"context": context, "question": state["original_query"]}
    ).content.strip()
    return {"answer": answer}


# --------------------------------------------------------------------------- #
# Build the graph
# --------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(RAGState)

    g.add_node("classify", classify_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("decompose", decompose_node)
    g.add_node("parallel_retrieve", parallel_retrieve_node)
    g.add_node("summarize", summarize_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("classify")
    g.add_conditional_edges(
        "classify",
        route_by_type,
        {"simple": "rewrite", "complex": "decompose"},
    )
    # Simple path
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "generate")
    # Complex path
    g.add_edge("decompose", "parallel_retrieve")
    g.add_edge("parallel_retrieve", "summarize")
    g.add_edge("summarize", "generate")

    g.add_edge("generate", END)
    return g.compile()


graph = build_graph()


def run_query(query: str) -> dict:
    initial: RAGState = {
        "original_query": query,
        "query_type": None,
        "rewritten_query": None,
        "sub_queries": [],
        "retrieved_chunks": [],
        "summaries": [],
        "answer": None,
    }
    final = graph.invoke(initial)
    return {
        "answer": final["answer"],
        "query_type": final["query_type"],
        "rewritten_query": final.get("rewritten_query"),
        "sub_queries": final.get("sub_queries", []),
        "sources": sorted({c["source"] for c in final["retrieved_chunks"]}),
    }


# --------------------------------------------------------------------------- #
# Optional FastAPI server
# --------------------------------------------------------------------------- #
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Hierarchical Hybrid RAG Backend")

    class QueryRequest(BaseModel):
        query: str

    @app.post("/query")
    def query_endpoint(req: QueryRequest):
        return run_query(req.query)

except ImportError:
    app = None  # FastAPI not installed; CLI still works


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Hierarchical Hybrid RAG — type a question (or 'exit'):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        result = run_query(q)
        print(f"\n[{result['query_type']}] {result['answer']}")
        if result["sources"]:
            print(f"\nSources: {', '.join(result['sources'])}")
