# Pre-Claim Denial Preventer: Clinical RAG Architecture

A neuro-symbolic proof-of-concept for evaluating medical necessity using structured (ICD-11) and unstructured (DSM-5, payer policies) data. The system is designed as an academic prototype for auditing clinical documentation against insurance criteria to estimate denial risk before submission.

## Quick Start (Zero-Wait Grader Setup)

To save grading time, the 40,000+ chunk database has been pre-computed. You do not need to run any data ingestion scripts.

1. **Download the Pre-built Database**
   Download `chroma_db.zip` from this [Google Drive link](https://drive.google.com/file/d/13gTANhkm259Yu6K0_EfSa_dp3p9K9PAc/view?usp=sharing).
   Extract the folder and place it in the repository so the path is exactly `data/chroma_db`.

2. **Set your Google API Key**
   Create a `.env` file in the root directory, or use `.env.example` as a template.
   Add your key as `GOOGLE_API_KEY=your_key_here`.

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run src/app.py
   ```

If `data/chroma_db` is missing, incomplete, or built from low-quality source files, retrieval quality will degrade sharply. This is especially noticeable for payer sources that depend on user-supplied uploads or image-heavy guideline documents.

## Architecture Overview

The system operates through a 4-phase clinical pipeline:

1. **Structural Ingestion**: Uses `PyMuPDF` to convert clinical PDFs to Markdown, then chunks on section-style boundaries to preserve payer rules and diagnostic references as coherent retrieval units.
2. **Vector Database**: Stores ICD-11, DSM-5, and payer-policy chunks in **ChromaDB**, using the `all-MiniLM-L6-v2` embedding model.
3. **Dynamic Source-Balanced Retrieval**: Prioritizes payer-specific rules while reserving a smaller share of retrieval volume for DSM-5 / ICD-11 baseline context, reducing the chance that general reference material drowns out policy criteria.
4. **Neuro-Symbolic Generation**: Uses **Gemini 2.5 Flash** as a utilization-review assistant. The prompt emphasizes policy-grounded reasoning, direct citations, and explicit disclosure when exact criteria are unavailable.

## Algorithmic Evaluation (Ragas)

This repository includes a small proof-of-concept Ragas evaluation over 3 hand-authored test cases. The current results should be interpreted as an early signal on retrieval and generation behavior, not as a definitive validation benchmark.

- **Faithfulness: 0.2233**: This relatively low score reflects the current trade-off in the prompt design. When exact payer criteria are missing, the model is allowed to provide a reasoned clinical estimate instead of stopping at "insufficient context."
- **Context Precision: 0.5278**: Retrieval can locate relevant payer content in some cases, but performance remains sensitive to corpus quality and document format.

### Architectural Trade-offs

- **Deductive Reasoning Over Strict Extraction**: The system is explicitly prompted to use clinical deduction when exact level-of-care rules are missing, trading strict source-text faithfulness for higher clinical utility in borderline or incomplete cases.
- **Degraded Context Failsafe**: The system includes a hard routing intercept for image-heavy or unparseable PDFs such as LOCUS-style guideline documents. When extraction returns placeholder text like `picture ... intentionally omitted`, the pipeline aborts normal generation and returns `[STATUS: WARNING]` instead of hallucinating policy logic. This improves safety, but it also lowers automated Ragas faithfulness scores.

Known limitations in the current prototype:

- **Small evaluation set**: The current automated evaluation covers only 3 cases, so the metrics are useful for debugging but not strong enough to support broad performance claims.
- **Image-heavy PDFs**: Some payer documents contain tables or scanned content that `pymupdf4llm` does not reliably convert into usable text. When that happens, retrieval quality drops and the app surfaces a data quality warning.
- **Prototype trade-off**: The system prefers transparent fallback reasoning over silence when exact criteria are missing, which improves usability for demos but can reduce faithfulness scores.

## Project Structure

```plaintext
|-- src/
|   |-- app.py                  # Streamlit UI: clinical form and output rendering
|   |-- ingest_icd.py           # Structured ICD-11 Excel parsing
|   |-- ingest_pdfs.py          # PDF-to-Markdown conversion and chunking
|   |-- initialize_db.py        # ChromaDB batch embedding and persistence
|   |-- retrieval.py            # Metadata filtering and source balancing
|   |-- generation.py           # Prompting and Gemini integration
|   `-- evaluate.py             # Ragas evaluation pipeline
|-- data/
|   |-- raw/                    # Source PDFs and Excel files
|   `-- chroma_db/              # Persisted vector database
|-- assets/                     # UI screenshots and documentation images
|-- .env.example                # Template for API keys
|-- requirements.txt            # Python dependencies
`-- README.md                   # Project documentation
```
