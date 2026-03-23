"""
Task 6: Prompt Engineering & LLM Integration
==============================================
Packages user inputs + retrieved context into a clinical auditor
prompt and sends it to the LLM for a structured audit report.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _HAS_GOOGLE = True
except ImportError:
    _HAS_GOOGLE = False
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval import ClinicalRetriever

# ── Load environment variables ──────────────────────────────────────
load_dotenv()

# ── System Prompt — Clinical Auditor Persona ────────────────────────
SYSTEM_PROMPT = """\
You are a Senior Clinical Utilization Reviewer with extensive experience
in behavioral health medical necessity determinations. Your role is to
evaluate whether a proposed treatment plan meets the insurance payer's
documented medical necessity criteria.

RULES:
1. CRITICAL: You must ONLY use the provided retrieved context. You are forbidden
   from making clinical assumptions or inferring medical necessity rules that
   are not explicitly written in the context. If the context is insufficient,
   state 'Insufficient Context to determine medical necessity'. Do NOT rely
   on your pre-trained knowledge of insurance policies.
2. When citing criteria, reference the exact [Source] and [Section] tags
   from the retrieved context.
3. If the insurance criteria do not support the requested treatment based
   on the documented symptoms/diagnosis, flag it as a "Likely Denial" and
   explain which specific criteria are not met.
4. If the documentation is missing a required element (e.g., evidence of
   failed outpatient therapy, risk of decompensation), specify EXACTLY
   what is missing and what the clinician should add.
5. Be objective and analytical. Never speculate beyond the provided context.

OUTPUT FORMAT — you MUST follow this exact structure:

## Audit Summary
A 2-3 sentence overview of the determination.

## Clinical Alignment
How the reported symptoms align (or don't) with DSM-5 / ICD-11
diagnostic criteria from the retrieved context.

## Insurance Criteria Match
Direct citations from the retrieved insurance policy chunks, referencing
specific section numbers and source documents.

## Missing Elements / Risks
Specific gaps in the clinical documentation that could lead to a denial.
List each gap as a bullet point with the exact criteria it fails to satisfy.

## Confidence Score
A single numeric value from 0-100% representing the estimated likelihood
of claim approval, followed by a one-sentence justification.
Example: **65%** — Documentation supports diagnosis but lacks evidence
of failed lower-level care.
"""

# ── Chat Prompt Template ────────────────────────────────────────────
AUDIT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """\
Please evaluate the following clinical case against the retrieved
insurance medical necessity criteria.

--- PATIENT INFORMATION ---
Professional Diagnosis: {diagnosis}
Symptoms: {symptoms}
Requested Treatment: {treatment}
Additional Notes: {additional_notes}

--- RETRIEVED CONTEXT ---
{retrieved_context}

Provide your structured audit report now.
"""),
])


def _init_llm():
    """
    Initialize the LLM from .env variables.
    Supports:
      - Google Gemini (if GOOGLE_API_KEY is set)
      - Any OpenAI-compatible endpoint (Together AI, Fireworks, Ollama, etc.)
    """
    # ── Check for Google Gemini first ───────────────────────────────
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key and google_key != "your_key_here":
        if not _HAS_GOOGLE:
            raise ImportError(
                "GOOGLE_API_KEY is set but langchain-google-genai is not "
                "installed. Run: py -m pip install langchain-google-genai"
            )
        model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")
        print(f"[LLM] Using Google Gemini: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_key,
            temperature=0.1,
            max_output_tokens=2048,
        )

    # ── Fall back to OpenAI-compatible endpoint ─────────────────────
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "No API key found. Set GOOGLE_API_KEY for Gemini, or "
            "LLM_API_KEY for an OpenAI-compatible provider."
        )

    print(f"[LLM] Using OpenAI-compatible: {model_name} @ {base_url}")
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.1,
        max_tokens=2048,
    )


def generate_audit_report(inputs: dict) -> str:
    """
    End-to-end pipeline: retrieve context → format prompt → call LLM.

    Parameters
    ----------
    inputs : dict
        Must contain keys: diagnosis, symptoms, treatment,
        insurance_provider. Optionally: additional_notes.

    Returns
    -------
    str
        The full audit report from the LLM.
    """
    # ── 1. Retrieve relevant context ────────────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_dir = os.path.join(project_root, "data", "chroma_db")

    retriever = ClinicalRetriever(persist_directory=persist_dir)

    docs = retriever.get_relevant_context(
        query=f"{inputs['diagnosis']} {inputs['symptoms']} {inputs['treatment']}",
        insurance_provider=inputs["insurance_provider"],
        top_k=8,  # More context for the LLM to work with
    )

    formatted_context = retriever.format_context(docs)

    # ── 2. Build the LLM chain ──────────────────────────────────────
    llm = _init_llm()
    chain = AUDIT_TEMPLATE | llm

    # ── 3. Generate the report ──────────────────────────────────────
    response = chain.invoke({
        "diagnosis": inputs["diagnosis"],
        "symptoms": inputs["symptoms"],
        "treatment": inputs["treatment"],
        "additional_notes": inputs.get("additional_notes", "None provided."),
        "retrieved_context": formatted_context,
    })

    return response.content


# ── Testing Block ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("HIGH-STAKES TEST CASE")
    print("=" * 70)

    test_inputs = {
        "diagnosis": "Major Depressive Disorder, Recurrent",
        "symptoms": (
            "Patient reports low mood and lethargy. No suicidal ideation. "
            "Currently in weekly therapy but feeling 'stuck'."
        ),
        "treatment": "Requesting 5-day-a-week Intensive Outpatient (IOP) program.",
        "insurance_provider": "HUSKY",
        "additional_notes": "Patient has been in outpatient therapy for 3 months.",
    }

    print(f"\nDiagnosis:  {test_inputs['diagnosis']}")
    print(f"Symptoms:   {test_inputs['symptoms']}")
    print(f"Treatment:  {test_inputs['treatment']}")
    print(f"Insurance:  {test_inputs['insurance_provider']}")
    print(f"Notes:      {test_inputs['additional_notes']}")
    print("\n" + "-" * 70)
    print("Generating audit report...\n")

    try:
        report = generate_audit_report(test_inputs)
        print(report)
    except ValueError as e:
        print(f"\n⚠ {e}")
        print("\nTo run this test, create a .env file with your API credentials:")
        print("  1. Copy .env.example to .env")
        print("  2. Set LLM_API_KEY to your provider's API key")
        print("  3. Set LLM_BASE_URL to the provider's endpoint")
        print("  4. Set LLM_MODEL_NAME to the model you want to use")
