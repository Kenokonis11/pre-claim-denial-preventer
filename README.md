# Pre-Claim Denial Preventer

A Retrieval-Augmented Generation (RAG) pipeline that helps clinicians validate medical claims before submission by cross-referencing clinical documentation against DSM-5, ICD-11, and insurance-specific medical necessity criteria.

## Overview

This system takes structured clinical inputs (diagnosis, symptoms, treatment, and insurance provider) and retrieves applicable rules from a vector database of medical and insurance policy documents. It then generates a confidence-scored assessment of whether a claim is likely to be approved or denied, along with billing context and reasoning.

## Architecture

- **Data Sources**: DSM-5, ICD-11 (structured), insurance medical necessity criteria (HUSKY/CT BHP, Anthem, Cigna, Health New England, LOCUS)
- **Chunking**: Structural/semantic chunking that preserves clinical rule boundaries
- **Embeddings**: HuggingFace open-source model (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB (local)
- **LLM**: Qwen 3.5 (or similar)
- **Evaluation**: Ragas framework (Faithfulness, Context Precision, Answer Relevance)

## Project Structure

```
pre-claim-denial-preventer/
├── data/               # Raw data files (DSM-5, ICD-11, insurance PDFs)
├── src/                # Source code
│   ├── ingestion/      # Data loading and chunking
│   ├── retrieval/      # Vector DB and retrieval logic
│   ├── generation/     # Prompt templates and LLM integration
│   └── ui/             # User interface
├── evals/              # Ragas evaluation scripts and test sets
├── requirements.txt    # Python dependencies
└── README.md
```

## Usage

*Setup and usage instructions will be added as the project is built.*

## Course Context

Built for a Generative AI course, demonstrating RAG pipeline construction, semantic chunking strategies, embedding-based retrieval, and rigorous LLM evaluation using the Ragas framework.
