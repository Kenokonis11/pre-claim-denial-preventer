"""
Task 3: PDF Structural Chunking
=================================
Converts insurance / clinical PDFs to Markdown via pymupdf4llm,
then chunks on section headers so that medical criteria stay intact.
"""

import os
import re
import pymupdf4llm
from langchain_core.documents import Document


# ── Map filename prefixes to insurance provider names ───────────────
_PROVIDER_MAP = {
    "anthem":  "Anthem",
    "cigna":   "Cigna",
    "ctbhp":   "HUSKY",
    "hne":     "Health New England",
    "other":   "LOCUS",
    "dsm":     "DSM-5",
}


def _detect_provider(filename: str) -> str:
    """Derive the insurance provider tag from the filename prefix."""
    lower = filename.lower()
    for prefix, provider in _PROVIDER_MAP.items():
        if lower.startswith(prefix):
            return provider
    return "Unknown"


# ── Regex for structural section boundaries ─────────────────────────
# Matches lines that look like section headers in clinical documents:
#   • Markdown headers:  ## **Clinical Indications**
#   • Alpha-numeric:     B.1.0 Admission Criteria
#   • Numbered sections: 3.2.1 Continued Stay
#   • Roman numerals:    IV. Documentation Requirements
_SECTION_RE = re.compile(
    r"^(?:"
    r"#{1,4}\s"                            # Markdown headers
    r"|[A-Z]\.\d+(?:\.\d+)*\s"            # A.1.0, B.2.3, etc.
    r"|\d+\.\d+(?:\.\d+)*\s"              # 1.2, 3.2.1, etc.
    r"|[IVXLC]+\.\s"                       # Roman numeral sections
    r")",
    re.MULTILINE,
)

# Max chunk size (characters). If a section exceeds this, we sub-split
# at paragraph boundaries so embeddings stay within model limits.
_MAX_CHUNK_SIZE = 2000
_MIN_CHUNK_SIZE = 100


def _extract_header(text: str) -> str:
    """Pull the first line of a chunk to use as the section label."""
    first_line = text.strip().split("\n", 1)[0]
    # Strip markdown formatting for a cleaner metadata label
    clean = re.sub(r"[#*]+", "", first_line).strip()
    return clean[:120]  # cap length for metadata


def _split_on_sections(markdown_text: str) -> list[str]:
    """
    Split markdown text at section boundaries identified by _SECTION_RE.
    Each resulting chunk starts with its section header.
    """
    # Find all section header positions
    split_points = [m.start() for m in _SECTION_RE.finditer(markdown_text)]

    if not split_points:
        # No headers detected — return the whole text as one chunk
        return [markdown_text.strip()] if markdown_text.strip() else []

    # Ensure we capture any preamble before the first header
    if split_points[0] > 0:
        split_points.insert(0, 0)

    chunks = []
    for i, start in enumerate(split_points):
        end = split_points[i + 1] if i + 1 < len(split_points) else len(markdown_text)
        chunk = markdown_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _subsplit_large_chunk(text: str) -> list[str]:
    """
    If a chunk exceeds _MAX_CHUNK_SIZE, split it at paragraph
    boundaries (double newlines) so that sentences are never cut mid-word.
    """
    if len(text) <= _MAX_CHUNK_SIZE:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    sub_chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > _MAX_CHUNK_SIZE and current:
            sub_chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        sub_chunks.append(current.strip())

    return sub_chunks


def _chunk_markdown(markdown_text: str) -> list[str]:
    """
    Two-pass chunking:
      1. Split on structural section headers.
      2. Sub-split oversized sections at paragraph boundaries.
    Drops any chunks below _MIN_CHUNK_SIZE (noise / page footers).
    """
    sections = _split_on_sections(markdown_text)

    final_chunks: list[str] = []
    for section in sections:
        for sub in _subsplit_large_chunk(section):
            if len(sub) >= _MIN_CHUNK_SIZE:
                final_chunks.append(sub)

    return final_chunks


def load_and_chunk_pdfs(directory_path: str) -> list[Document]:
    """
    Iterate through all .pdf files in *directory_path*, convert each to
    Markdown, structurally chunk them, and return a flat list of
    LangChain Document objects with source metadata.
    """
    pdf_files = sorted(
        f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        print(f"[PDF] No PDF files found in {directory_path}")
        return []

    print(f"[PDF] Found {len(pdf_files)} PDF files to process")

    all_documents: list[Document] = []

    for filename in pdf_files:
        filepath = os.path.join(directory_path, filename)
        provider = _detect_provider(filename)

        print(f"  → Processing: {filename}  (provider: {provider})")

        # ── Convert PDF → Markdown ──────────────────────────────────
        try:
            markdown_text = pymupdf4llm.to_markdown(filepath)
        except Exception as e:
            print(f"    ⚠ Failed to parse {filename}: {e}")
            continue

        # ── Structural chunking ─────────────────────────────────────
        chunks = _chunk_markdown(markdown_text)
        print(f"    ✓ {len(chunks)} chunks created")

        for chunk_text in chunks:
            section_header = _extract_header(chunk_text)

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": filename,
                    "insurance_provider": provider,
                    "section": section_header,
                },
            )
            all_documents.append(doc)

    print(f"\n[PDF] Total Documents created: {len(all_documents)}")
    return all_documents


# ── Testing Block ───────────────────────────────────────────────────
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "..", "data", "raw")

    if not os.path.isdir(raw_dir):
        print(f"[ERROR] Directory not found: {os.path.abspath(raw_dir)}")
        exit(1)

    docs = load_and_chunk_pdfs(raw_dir)

    # ── Print 3 consecutive chunks from an insurance PDF ────────────
    # Find the first chunk that belongs to an insurance provider
    # (skip DSM-5 for this preview since it's massive)
    insurance_docs = [
        d for d in docs if d.metadata["insurance_provider"] != "DSM-5"
    ]

    if len(insurance_docs) >= 3:
        # Pick documents from the middle to show real criteria, not cover pages
        mid = len(insurance_docs) // 3
        sample = insurance_docs[mid : mid + 3]
    else:
        sample = insurance_docs[:3]

    print("\n" + "=" * 70)
    print("SAMPLE: 3 consecutive insurance chunks")
    print("=" * 70)
    for i, doc in enumerate(sample, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"  metadata: {doc.metadata}")
        # Show first 500 chars of content to keep output readable
        preview = doc.page_content[:500]
        if len(doc.page_content) > 500:
            preview += "\n  [...truncated...]"
        print(f"  page_content:\n{preview}")
        print()
