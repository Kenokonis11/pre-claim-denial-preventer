import os
import sys
from functools import lru_cache

from datasets import Dataset
from dotenv import load_dotenv
from google import genai
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._faithfulness import Faithfulness

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.generation import generate_audit_report, get_cached_llm, get_cached_retriever

load_dotenv()


def _require_eval_dependencies():
    try:
        import jsonref  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing evaluation dependency: jsonref. "
            "Run `pip install -r requirements.txt` or `pip install jsonref` "
            "before running `py src/evaluate.py`."
        ) from exc


@lru_cache(maxsize=1)
def get_ragas_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    llm_model = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)
    return llm_factory(
        llm_model,
        provider="google",
        client=client,
        temperature=0.0,
    )


def main():
    _require_eval_dependencies()

    test_cases = [
        {
            "diagnosis": "Major Depressive Disorder, Severe",
            "symptoms": "Patient presents with severe lethargy, daily suicidal ideation without a specific plan, and severe functional impairment at work. Patient has high risk of decompensation without structured daily care.",
            "treatment": "Intensive Outpatient Program (IOP)",
            "insurance_provider": "HUSKY",
            "ground_truth": "Likely Approval. The patient meets criteria for IOP due to severe functional impairment and high risk of decompensation requiring structured care.",
        },
        {
            "diagnosis": "Autism Spectrum Disorder",
            "symptoms": "Patient requires assistance with social skills. No history of trying lower-level routine outpatient therapy.",
            "treatment": "Intensive Outpatient Program (IOP)",
            "insurance_provider": "Cigna",
            "ground_truth": "Likely Denial. The documentation is missing evidence of a failure of lower-level routine outpatient care prior to requesting IOP.",
        },
        {
            "diagnosis": "Generalized Anxiety Disorder",
            "symptoms": "Moderate anxiety, poor recovery environment at home, but highly engaged in treatment and no medical comorbidities.",
            "treatment": "Partial Hospitalization Program (PHP)",
            "insurance_provider": "LOCUS",
            "ground_truth": "Likely Denial or Step-Down. Based on LOCUS dimensions, the patient's moderate symptoms and high engagement do not justify the intensity of PHP.",
        },
    ]

    persist_dir = os.path.join(project_root, "data", "chroma_db")
    retriever = get_cached_retriever(persist_dir)
    generation_llm = get_cached_llm()
    ragas_llm = get_ragas_llm()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Generating Pipeline Outputs for the 3 Test Cases...")
    for idx, case in enumerate(test_cases, 1):
        print(f"  [{idx}/3] Processing: {case['insurance_provider']}")

        question = (
            f"Diagnosis: {case['diagnosis']}, Symptoms: {case['symptoms']}, "
            f"Treatment: {case['treatment']}, Provider: {case['insurance_provider']}"
        )
        query = f"{case['diagnosis']} {case['symptoms']} {case['treatment']}"

        raw_docs = retriever.get_relevant_context(
            query=query,
            insurance_provider=case["insurance_provider"],
            top_k=8,
        )
        case_contexts = [doc.page_content for doc in raw_docs]

        inputs = {
            "diagnosis": case["diagnosis"],
            "symptoms": case["symptoms"],
            "treatment": case["treatment"],
            "insurance_provider": case["insurance_provider"],
            "additional_notes": "None provided.",
        }
        answer = generate_audit_report(inputs, retriever=retriever, llm=generation_llm)

        questions.append(question)
        answers.append(answer)
        contexts.append(case_contexts)
        ground_truths.append(case["ground_truth"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths,
        }
    )

    print("\nStarting Ragas Evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ContextPrecision(),
        ],
        llm=ragas_llm,
    )

    print("\n=======================================================")
    print("FINAL RAGAS EVALUATION METRICS")
    print("=======================================================")
    print(result)

    csv_path = os.path.join(project_root, "evaluation_results.csv")
    try:
        df = result.to_pandas()
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed evaluation results saved to: {csv_path}")
    except Exception as e:
        print(f"\nCould not save CSV file: {e}")


if __name__ == "__main__":
    main()
