import json
import os
import re
import TypedDict

from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from fastembed import FastEmbed

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from operator import add

from qdrant_client import QdrantClient, models

from sentence_transformers import CrossEncoder, SentenceTransformer

from typing import List

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

DENSE_MODEL_NAME = os.getenv("DENSE_MODEL","")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL","")
CROSS_ENCODER_NAME = os.getenv("CROSS_ENCODER_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2")
                             
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
            "parent_text":p.payload.get("parent_text",p.payload["text"]),
            "source":p.payload.get("source","unknown"),
            "score":p.score
        }
        for p in query_results.points
    ]