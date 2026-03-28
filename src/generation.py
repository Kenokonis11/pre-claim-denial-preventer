"""
Task 6: Prompt Engineering & LLM Integration
==============================================
Packages user inputs + retrieved context into a clinical auditor
prompt and sends it to the LLM for a structured audit report.
"""

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    _HAS_GOOGLE = True
except ImportError:
    _HAS_GOOGLE = False

from src.retrieval import ClinicalRetriever

load_dotenv()

SYSTEM_PROMPT = """\
You are a Senior Clinical Utilization Reviewer with extensive experience
in behavioral health medical necessity determinations. Your role is to
evaluate whether a proposed treatment plan meets the insurance payer's
documented medical necessity criteria.

RULES:
1. If the retrieved context lacks the specific medical necessity criteria for
   the requested treatment, you must NOT simply state 'Insufficient Context' and stop.
   Instead, you must follow a strict two-step process:
   - Explicit Disclaimer: You must begin your assessment by explicitly stating:
     'Disclaimer: The exact medical necessity criteria for the requested treatment
     were not found in the retrieved insurance guidelines.'
   - Reasoned Clinical Estimate: After the disclaimer, you must use clinical deduction
     to provide a best estimate. Evaluate the patient's symptoms against the
     insurance criteria provided for the other levels of care present in the
     context (e.g., Inpatient, ECT, IOP, etc.). Deduce and explain whether the
     requested treatment is likely appropriate, insufficient, or excessive by
     comparing the patient's clinical severity to the thresholds of the rules
     you do possess.
2. When citing criteria, reference the exact [Source] and [Section] tags from
   the retrieved context.
3. Primary Evidence: You must still prioritize using the provided retrieved context
   for clinical baseline (DSM-5/ICD-11) and any other available insurance rules.
4. If the documentation is missing a required element based on the deduced thresholds,
   specify EXACTLY what is missing and what the clinician should add.
5. Be objective and analytical. While deduction is required when criteria are missing,
   it must be based on the relationship between levels of care in the context.

OUTPUT FORMAT:
You must begin your response with exactly one of the following tags on its own line:
[STATUS: APPROVAL], [STATUS: DENIAL], or [STATUS: WARNING]. After the tag, leave
a blank line, and then write your comprehensive Markdown clinical audit report.

The report should follow this markdown structure:
## Audit Summary
A 2-3 sentence overview of the determination.

## Clinical Alignment
How the reported symptoms align (or don't) with DSM-5 / ICD-11 diagnostic criteria
from the retrieved context.

## Insurance Criteria Match
Direct citations from the retrieved insurance policy chunks, referencing specific
section numbers and source documents.

## Missing Elements / Risks
Specific gaps in the clinical documentation that could lead to a denial.

## Confidence Score
A single numeric value from 0-100% and justification.
"""

AUDIT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """\
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
""",
        ),
    ]
)

_IMAGE_PLACEHOLDER_RE = re.compile(
    r"(picture\s*\[[^\]]+\]\s*intentionally omitted|==>\s*picture\s*\[[^\]]+\]\s*intentionally omitted\s*<==)",
    re.IGNORECASE,
)
_DEGRADED_WARNING = (
    "[STATUS: WARNING]\n\n"
    "Data Quality Warning: Retrieved policy documents contain unparseable images "
    "or tables. Clinical deduction cannot be safely performed."
)


@lru_cache(maxsize=1)
def get_cached_llm():
    """
    Initialize the LLM once per process.
    Supports Google Gemini or any OpenAI-compatible endpoint.
    """
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
            max_output_tokens=4096,
        )

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


@lru_cache(maxsize=4)
def get_cached_retriever(persist_dir: str) -> ClinicalRetriever:
    """Load the embedding model and Chroma client once per persist directory."""
    return ClinicalRetriever(persist_directory=persist_dir)


def _is_heavily_degraded_context(retrieved_context: str) -> bool:
    """
    Detect contexts dominated by placeholder text from image/table extraction failures.
    """
    placeholder_count = len(_IMAGE_PLACEHOLDER_RE.findall(retrieved_context))
    if placeholder_count >= 3:
        return True

    lines = [line.strip() for line in retrieved_context.splitlines() if line.strip()]
    if not lines:
        return True

    placeholder_lines = [
        line for line in lines if _IMAGE_PLACEHOLDER_RE.search(line)
    ]
    return len(placeholder_lines) / max(1, len(lines)) >= 0.35


def generate_audit_report(
    inputs: dict,
    retriever: ClinicalRetriever | None = None,
    llm=None,
) -> str:
    """
    End-to-end pipeline: retrieve context -> validate context quality -> call LLM.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_dir = os.path.join(project_root, "data", "chroma_db")

    retriever = retriever or get_cached_retriever(persist_dir)

    docs = retriever.get_relevant_context(
        query=f"{inputs['diagnosis']} {inputs['symptoms']} {inputs['treatment']}",
        insurance_provider=inputs["insurance_provider"],
        top_k=8,
    )

    formatted_context = retriever.format_context(docs)
    if _is_heavily_degraded_context(formatted_context):
        return _DEGRADED_WARNING

    llm = llm or get_cached_llm()
    chain = AUDIT_TEMPLATE | llm

    response = chain.invoke(
        {
            "diagnosis": inputs["diagnosis"],
            "symptoms": inputs["symptoms"],
            "treatment": inputs["treatment"],
            "additional_notes": inputs.get("additional_notes", "None provided."),
            "retrieved_context": formatted_context,
        }
    )

    return response.content.strip()


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
        print(f"\nWarning: {e}")
        print("\nTo run this test, create a .env file with your API credentials:")
        print("  1. Copy .env.example to .env")
        print("  2. Set LLM_API_KEY to your provider's API key")
        print("  3. Set LLM_BASE_URL to the provider's endpoint")
        print("  4. Set LLM_MODEL_NAME to the model you want to use")
