"""
Task 4: Vector Database Initialization
========================================
Embeds all Document objects (ICD-11 + PDFs) into a local ChromaDB
instance using HuggingFace all-MiniLM-L6-v2 embeddings.
"""

import os
import sys
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Import our ingestion modules ────────────────────────────────────
from src.ingest_icd import load_icd_excel
from src.ingest_pdfs import load_and_chunk_pdfs


# ── Constants ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 500  # Documents per batch to avoid memory spikes


def initialize_vector_db(
    icd_path: str,
    pdf_dir: str,
    persist_directory: str = "data/chroma_db",
) -> Chroma:
    """
    Build (or rebuild) the ChromaDB vector store from all ingested
    documents.  Processes documents in batches to prevent OOM crashes
    on the ~40,000-document corpus.

    Parameters
    ----------
    icd_path : str
        Path to the ICD-11 Excel file.
    pdf_dir : str
        Path to the directory containing clinical PDFs.
    persist_directory : str
        Where to store the ChromaDB files on disk.

    Returns
    -------
    Chroma
        The initialised (and persisted) vector store.
    """

    # ── 1. Load all documents ───────────────────────────────────────
    print("=" * 60)
    print("PHASE 1: Loading documents")
    print("=" * 60)

    icd_docs = load_icd_excel(icd_path)
    pdf_docs = load_and_chunk_pdfs(pdf_dir)

    all_docs = icd_docs + pdf_docs
    total = len(all_docs)
    print(f"\n[DB] Combined corpus: {total} documents "
          f"({len(icd_docs)} ICD-11 + {len(pdf_docs)} PDF chunks)")

    # ── 2. Initialize the embedding model ───────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Loading embedding model")
    print("=" * 60)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[DB] Embedding model loaded: {EMBEDDING_MODEL}")

    # ── 3. Build ChromaDB in batches ────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 3: Embedding & storing ({total} docs in batches of {BATCH_SIZE})")
    print("=" * 60)

    # Clear any existing DB so we get a clean build
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
        print(f"[DB] Cleared existing database at {persist_directory}")

    os.makedirs(persist_directory, exist_ok=True)

    start_time = time.time()
    db = None

    for i in range(0, total, BATCH_SIZE):
        batch = all_docs[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        if db is None:
            # First batch — create the collection
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name="clinical_docs",
            )
        else:
            # Subsequent batches — add to existing collection
            db.add_documents(batch)

        elapsed = time.time() - start_time
        rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
        eta = (total - i - len(batch)) / rate if rate > 0 else 0

        print(
            f"  Batch {batch_num}/{total_batches} complete  "
            f"({i + len(batch)}/{total} docs)  "
            f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
        )

    elapsed_total = time.time() - start_time
    print(f"\n[DB] ✓ Database built in {elapsed_total:.1f}s")
    print(f"[DB] Persisted to: {os.path.abspath(persist_directory)}")

    return db


# ── Testing Block ───────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Resolve paths ───────────────────────────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    icd_path = os.path.join(project_root, "data", "raw", "icd11.xlsx")
    pdf_dir = os.path.join(project_root, "data", "raw")
    persist_dir = os.path.join(project_root, "data", "chroma_db")

    # ── Build the database ──────────────────────────────────────────
    db = initialize_vector_db(icd_path, pdf_dir, persist_dir)

    # ── Similarity Search Test ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("SIMILARITY SEARCH TEST: 'Major Depressive Disorder'")
    print("=" * 60)

    results = db.similarity_search("Major Depressive Disorder", k=3)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"  source: {doc.metadata.get('source', 'N/A')}")
        print(f"  metadata: {doc.metadata}")
        preview = doc.page_content[:300]
        if len(doc.page_content) > 300:
            preview += "\n  [...truncated...]"
        print(f"  page_content:\n  {preview}")
