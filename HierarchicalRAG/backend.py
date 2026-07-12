import json
import os
import re
from typing import TypedDict, Optional, List, Literal, Annotated

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from operator import add

from qdrant_client import QdrantClient, models

from sentence_transformers import CrossEncoder, SentenceTransformer
from fastembed import SparseTextEmbedding
from typing import List

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "hierarchical_rag")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
 
DENSE_MODEL_NAME = os.getenv("DENSE_MODEL","BAAI/bge-small-en-v1.5")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL","")
CROSS_ENCODER_NAME = os.getenv("CROSS_ENCODER_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2")
                             
MAX_PARALLEL_WORKERS=10
SLM_MODEL = os.getenv("SLM_MODEL","llama-3.1-8b-instant")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

slm = ChatGroq(
    model=SLM_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY
)
llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0.7,
    api_key=GROQ_API_KEY
)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

dense_model = SentenceTransformer(DENSE_MODEL_NAME)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def hybrid_search(query:str, top_k:int=5, sparse_weight:float=0.5, dense_weight:float=0.5):
    """
    Perform a hybrid search using both sparse and dense embeddings.

    Args:
        query (str): Query provided by the user.
        top_k (int): The number of top chunks to be fetched from the DB, according to the semantic and sparse search for BM_25 and Dense.
        sparse_weight (float): The weight for the sparse search results, to be used for generating the final score according to which the results are gonna be ranked.
        dense_weight (float): The weight for the dense search results, to be used for generating the final score according to which the results are gonna be ranked.
    """
    dense_vector = dense_model.encode(query, normalize_embeddings=True).tolist()
    # It normalizes the embeddings because it speeds up the calculation of the cosine similarity, because it doesn't have to do anything with magnitude of vectors
    sparse_emb = next(iter(sparse_model.embed([query])))
    sparse_vector = models.SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.to_list()
    )
    query_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=10),
            models.Prefetch(query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=10)
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True
    )

    return [{
            "text":p.payload["text"],
            "parent_id":p.payload["parent_id"],
            "parent_text":p.payload.get("parent_text",p.payload["text"]),
            "source":p.payload.get("source","unknown"),
            "score":p.score
        }
        for p in query_results.points
    ]

def rerank(query:str ,chunks:List[dict], top_k:int=5):

    if not chunks or not query:
        return []
    
    scores = cross_encoder.predict([(query, c["text"]) for c in chunks])

    for c, s in zip(chunks, scores):
        c["rerank_score"]=float(s)

    ranked = sorted(chunks, key=lambda c:c["rerank_score"], reverse=True)[:top_k]

    seen, unique = set(), []

    for c in ranked:
        key=c["parent_id"]
        if key not in seen:
           seen.add(key)
           unique.add(key)

def retrieve_and_rerank(query: str) -> List[dict]:
    return rerank(query, hybrid_search(query))

class RAGState(TypedDict):
    original_query: str
    query_type: Optional[Literal["simple", "complex"]]
    rewritten_query: Optional[str]
    sub_queries: List[str]
    retrieved_chunks: Annotated[List[dict], add]
    summaries: List[str]
    answer: Optional[str]

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

def route_by_type(state: RAGState)-> str:
    return state["query_by_type"]

def rewrite_node(state: RAGState) -> dict:
    """SIMPLE path: SLM expands/rewrites the query for better retrieval."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user's search query to maximize retrieval quality. "
                "Expand abbreviations, add key synonyms, make implicit intent explicit. "
                "Keep it a simple query. Output ONLY the rewritten query, nothing else.",
            ),
            ("human","{query}"),
        ]
    )

    rewritten = (prompt|slm).invoke({"query":state["original_query"]}).content.strip()
    print(f"[rewrite] {rewritten}")
    return {"rewritten_query":rewritten}

def retrieve_node(state:RAGState)-> dict:
    query=state["rewritten_query"] or state["original_query"]
    chunks = retrieve_and_rerank(query)

    print(f"[retrieve] {len(chunks)} chunks")

    return {"rewritten_query":chunks}

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

# done
def parallel_retrieve_node(state: RAGState) -> dict:

    with ThreadPoolExecutor(max_workers=10) as pool:
        results=list(pool.map(retrieve_and_rerank, state["sub_queries"]))

    all_chunks, seen= [],set()

    for sub_q, chunks in zip(state["sub_queries"],results):
        for c in chunks:
            key=c["parent_id"]
            if key not in seen:
                seen.add(key)
                c["sub_query"]=sub_q
                all_chunks.append(c)

    print(f'"[parallel_retrieve]" {len(all_chunks)}')
    return {"retrieved_chunks": all_chunks}

def generate_node(state: RAGState) -> dict:
    if state["query_type"]=='complex':
        context="\n\n".join(state["summaries"])
    else:
        context="\n\n---\n\n".join(c["parent_text"] for c in state["retrieved_chunks"])

    prompt= ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise assistant. Answer the user's question using ONLY "
                "the provided context. If the context is insufficient, say so. "
                "Cite sources inline when useful.",
            ),
            ("human","Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    answer= (prompt|llm).invoke({"context":context, "question":state["original_query"]}).content.strip()
    return {
        "answer":answer
    }