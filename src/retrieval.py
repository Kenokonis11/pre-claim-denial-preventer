"""
Task 5: Dynamic Retrieval Logic
=================================
Routes queries through the ChromaDB vector store with metadata
filtering so that only the selected insurance provider's rules
(plus universal DSM-5 / ICD-11 references) are retrieved.
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Map UI dropdown labels → metadata values used during ingestion
PROVIDER_LABELS = {
    "HUSKY":              "HUSKY",
    "Anthem":             "Anthem",
    "Cigna":              "Cigna",
    "Health New England": "Health New England",
    "General/Not Listed": "LOCUS",
}


class ClinicalRetriever:
    """
    Wraps the ChromaDB vector store with metadata-aware retrieval.

    Every query automatically includes DSM-5 and ICD-11 context.
    Insurance-specific chunks are filtered to match the user's
    dropdown selection, preventing "policy bleed" across providers.
    """

    def __init__(self, persist_directory: str = "data/chroma_db"):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self._embeddings,
            collection_name="clinical_docs",
        )
        print(f"[Retriever] Loaded ChromaDB from {persist_directory}")
        print(f"[Retriever] Collection size: {self._db._collection.count()}")

    # ── Primary retrieval method ────────────────────────────────────
    def get_relevant_context(
        self,
        query: str,
        insurance_provider: str,
        top_k: int = 5,
    ) -> list[Document]:
        """
        Retrieve the most relevant documents for *query*, filtered to:
          • ICD-11 rows       (always included)
          • DSM-5 chunks      (always included)
          • The selected insurance provider's chunks only

        Parameters
        ----------
        query : str
            The clinical search query (built from diagnosis + symptoms + treatment).
        insurance_provider : str
            One of the keys in PROVIDER_LABELS, or the raw metadata value.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[Document]
        """
        # Resolve UI label → metadata value
        provider_value = PROVIDER_LABELS.get(insurance_provider, insurance_provider)

        # Force top_k=4 for insurance_provider and top_k=2 for DSM-5/ICD-11
        insurance_results = self._db.similarity_search(
            query=query,
            k=4,
            filter={"insurance_provider": provider_value},
        )
        
        general_results = self._db.similarity_search(
            query=query,
            k=2,
            filter={
                "$or": [
                    {"source": "ICD-11"},
                    {"insurance_provider": "DSM-5"}
                ]
            }
        )

        return insurance_results + general_results

    # ── Context formatter for the LLM ───────────────────────────────
    @staticmethod
    def format_context(documents: list[Document]) -> str:
        """
        Convert retrieved Documents into a labelled string the LLM
        can reference for citations.

        Format per chunk:
            [Source: <filename> | Section: <header>]
            <content>
        """
        if not documents:
            return "(No relevant context found.)"

        sections: list[str] = []
        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            source = meta.get("source", "Unknown")
            section = meta.get("section", meta.get("code", "N/A"))
            provider = meta.get("insurance_provider", "")

            header = f"[Source: {source}"
            if provider:
                header += f" | Provider: {provider}"
            header += f" | Section: {section}]"

            sections.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(sections)


# ── Testing Block ───────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_dir = os.path.join(project_root, "data", "chroma_db")

    retriever = ClinicalRetriever(persist_directory=persist_dir)

    # ── Test 1: HUSKY + IOP ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: 'Intensive Outpatient Program criteria'  |  Provider: HUSKY")
    print("=" * 70)

    docs1 = retriever.get_relevant_context(
        query="Intensive Outpatient Program criteria",
        insurance_provider="HUSKY",
        top_k=5,
    )
    formatted1 = retriever.format_context(docs1)
    print(formatted1)

    # Show source breakdown
    sources1 = [d.metadata.get("insurance_provider", d.metadata.get("source")) for d in docs1]
    print(f"\n→ Source breakdown: {sources1}")

    # ── Test 2: Cigna + ASD ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: 'Autism Spectrum Disorder diagnostic codes'  |  Provider: Cigna")
    print("=" * 70)

    docs2 = retriever.get_relevant_context(
        query="Autism Spectrum Disorder diagnostic codes",
        insurance_provider="Cigna",
        top_k=5,
    )
    formatted2 = retriever.format_context(docs2)
    print(formatted2)

    sources2 = [d.metadata.get("insurance_provider", d.metadata.get("source")) for d in docs2]
    print(f"\n→ Source breakdown: {sources2}")
