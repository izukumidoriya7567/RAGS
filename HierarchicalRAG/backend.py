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
class RAGState(TypedDict):
    query_type: str
    original_query: str
    rewritten_query: str # the query after rewriting it by SLM
    sub_queries: List[str] # when the query is complex, it can be broken down into multiple sub-queries 

def build_graph():
    graph = StateGraph()

    graph.add
def rerank(query:str , chunks:List[dict], top_k_chunks:int=3) ->List[dict]:
    if not chunks or not query:
        return []

