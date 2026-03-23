"""
Task 2: ICD-11 Structured Data Ingestion
=========================================
Loads the ICD-11 Excel spreadsheet and converts rows into
LangChain Document objects with source metadata for retrieval filtering.
"""

import os
import pandas as pd
from langchain_core.documents import Document


def load_icd_excel(
    file_path: str,
    code_col: str | None = None,
    desc_col: str | None = None,
) -> list[Document]:
    """
    Read an ICD-11 Excel export and return LangChain Documents.

    Parameters
    ----------
    file_path : str
        Path to the .xlsx file.
    code_col : str, optional
        Column name for the ICD-11 code. If None, auto-detects from
        common WHO export headers (Code, BlockId, LinearizationURI, etc.).
    desc_col : str, optional
        Column name for the description. If None, auto-detects from
        common WHO export headers (Title, ClassKind, etc.).

    Returns
    -------
    list[Document]
        One Document per row with page_content and metadata.
    """

    # ── Load the spreadsheet ────────────────────────────────────────
    df = pd.read_excel(file_path, engine="openpyxl")

    # ── Auto-detect code column ─────────────────────────────────────
    CODE_CANDIDATES = ["Code", "BlockId", "LinearizationURI", "Linearization (MMS) URI", "code"]
    DESC_CANDIDATES = ["Title", "Description", "ClassKind", "title", "description"]

    if code_col is None:
        for candidate in CODE_CANDIDATES:
            if candidate in df.columns:
                code_col = candidate
                break
        if code_col is None:
            raise ValueError(
                f"Could not auto-detect code column. "
                f"Available columns: {list(df.columns)}. "
                f"Pass code_col= explicitly."
            )

    if desc_col is None:
        for candidate in DESC_CANDIDATES:
            if candidate in df.columns:
                desc_col = candidate
                break
        if desc_col is None:
            raise ValueError(
                f"Could not auto-detect description column. "
                f"Available columns: {list(df.columns)}. "
                f"Pass desc_col= explicitly."
            )

    print(f"[ICD-11] Using code_col='{code_col}', desc_col='{desc_col}'")
    print(f"[ICD-11] Total rows in spreadsheet: {len(df)}")

    # ── Clean NaN values ────────────────────────────────────────────
    df[code_col] = df[code_col].fillna("").astype(str).str.strip()
    df[desc_col] = df[desc_col].fillna("").astype(str).str.strip()

    # ── Drop rows that have no code AND no description ──────────────
    df = df[~((df[code_col] == "") & (df[desc_col] == ""))]

    # ── Convert to LangChain Documents ──────────────────────────────
    documents: list[Document] = []
    for _, row in df.iterrows():
        code = row[code_col]
        description = row[desc_col]

        doc = Document(
            page_content=f"ICD-11 Code: {code} - Description: {description}",
            metadata={
                "source": "ICD-11",
                "code": code,
            },
        )
        documents.append(doc)

    print(f"[ICD-11] Created {len(documents)} Document objects")
    return documents


# ── Testing Block ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Resolve path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, "..", "data", "raw", "icd11.xlsx")

    if not os.path.exists(default_path):
        print(f"[ERROR] File not found: {os.path.abspath(default_path)}")
        print("  Please copy / rename your ICD-11 Excel file to data/raw/icd11.xlsx")
        exit(1)

    docs = load_icd_excel(default_path)

    print("\n--- First 5 Documents ---")
    for doc in docs[:5]:
        print(f"  page_content: {doc.page_content}")
        print(f"  metadata:     {doc.metadata}")
        print()
