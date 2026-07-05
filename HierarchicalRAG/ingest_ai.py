"""
ingest_ai.py
============
Hierarchical ingestion pipeline for a Hybrid-Search RAG system.

- Hierarchical chunking: parent chunks (context) -> child chunks (retrieval units)
- Dense embeddings  : sentence-transformers (BAAI/bge-small-en-v1.5 by default)
- Sparse embeddings : BM25 (via fastembed's Qdrant/bm25 sparse model)
- Vector store      : Qdrant Cloud (named dense vector + named sparse vector)

Usage:
    export QDRANT_URL="https://xxxx.cloud.qdrant.io"
    export QDRANT_API_KEY="..."
    python ingest_ai.py --data_dir ./docs
"""

import argparse
import os
import uuid
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding  # pip install fastembed

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "hierarchical_rag")

DENSE_MODEL_NAME = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

BATCH_SIZE = 64


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_documents(data_dir: str) -> List[Document]:
    """Load .txt/.md and .pdf files from a directory."""
    docs: List[Document] = []

    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    md_loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    docs.extend(txt_loader.load())
    docs.extend(md_loader.load())

    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(os.path.join(root, f)).load())

    print(f"[load] Loaded {len(docs)} raw documents from {data_dir}")
    return docs


# --------------------------------------------------------------------------- #
# Hierarchical chunking
# --------------------------------------------------------------------------- #
def hierarchical_chunk(docs: List[Document]) -> List[Document]:
    """
    Split documents into parent chunks, then split each parent into child chunks.
    Child chunks are what we embed & retrieve; each child carries its parent's
    full text in metadata so the generator LLM gets wide context.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    children: List[Document] = []
    parents = parent_splitter.split_documents(docs)
    print(f"[chunk] {len(parents)} parent chunks")

    for parent in parents:
        parent_id = str(uuid.uuid4())
        for child in child_splitter.split_documents([parent]):
            child.metadata.update(
                {
                    "parent_id": parent_id,
                    "parent_text": parent.page_content,
                    "source": parent.metadata.get("source", "unknown"),
                }
            )
            children.append(child)

    print(f"[chunk] {len(children)} child chunks")
    return children


# --------------------------------------------------------------------------- #
# Qdrant collection setup
# --------------------------------------------------------------------------- #
def get_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY env vars.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


def ensure_collection(client: QdrantClient, dense_dim: int) -> None:
    if client.collection_exists(COLLECTION_NAME):
        print(f"[qdrant] Collection '{COLLECTION_NAME}' already exists")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dense_dim, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF  # BM25-style IDF weighting server-side
            )
        },
    )
    print(f"[qdrant] Created collection '{COLLECTION_NAME}' (dense dim={dense_dim})")


# --------------------------------------------------------------------------- #
# Embedding + upsert
# --------------------------------------------------------------------------- #
def ingest(children: List[Document]) -> None:
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    dense_dim = dense_model.get_sentence_embedding_dimension()

    client = get_client()
    ensure_collection(client, dense_dim)

    texts = [c.page_content for c in children]

    for start in range(0, len(children), BATCH_SIZE):
        batch_docs = children[start : start + BATCH_SIZE]
        batch_texts = texts[start : start + BATCH_SIZE]

        dense_vecs = dense_model.encode(
            batch_texts, normalize_embeddings=True, show_progress_bar=False
        )
        sparse_vecs = list(sparse_model.embed(batch_texts))

        points = []
        for doc, dv, sv in zip(batch_docs, dense_vecs, sparse_vecs):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        DENSE_VECTOR_NAME: dv.tolist(),
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sv.indices.tolist(),
                            values=sv.values.tolist(),
                        ),
                    },
                    payload={
                        "text": doc.page_content,
                        "parent_id": doc.metadata["parent_id"],
                        "parent_text": doc.metadata["parent_text"],
                        "source": doc.metadata.get("source", "unknown"),
                    },
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[ingest] Upserted {start + len(points)}/{len(children)} points")

    print("[ingest] Done ✅")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant (hybrid).")
    parser.add_argument("--data_dir", type=str, default="./docs", help="Folder of documents")
    args = parser.parse_args()

    raw_docs = load_documents(args.data_dir)
    if not raw_docs:
        raise SystemExit(f"No documents found in {args.data_dir}")

    child_chunks = hierarchical_chunk(raw_docs)
    ingest(child_chunks)
