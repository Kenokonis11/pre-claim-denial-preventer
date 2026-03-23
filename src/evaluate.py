import os
import sys
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from ragas import evaluate
from ragas.metrics import Faithfulness, ContextPrecision

from ragas.embeddings import GoogleEmbeddings
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from src.retrieval import ClinicalRetriever
from src.generation import generate_audit_report

load_dotenv()

def main():
    llm_model = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Ragas 0.4.3+ LLM Judge using LangChain fallback
    llm_judge = ChatGoogleGenerativeAI(
        model=llm_model,
        google_api_key=api_key,
        temperature=0.0
    )
    ragas_llm = LangchainLLMWrapper(llm_judge)

    # Ragas 0.4.3+ Embeddings
    ragas_embeddings = GoogleEmbeddings(api_key=api_key)

    test_cases = [
        {
            "diagnosis": "Major Depressive Disorder, Severe",
            "symptoms": "Patient presents with severe lethargy, daily suicidal ideation without a specific plan, and severe functional impairment at work. Patient has high risk of decompensation without structured daily care.",
            "treatment": "Intensive Outpatient Program (IOP)",
            "insurance_provider": "HUSKY",
            "ground_truth": "Likely Approval. The patient meets criteria for IOP due to severe functional impairment and high risk of decompensation requiring structured care."
        },
        {
            "diagnosis": "Autism Spectrum Disorder",
            "symptoms": "Patient requires assistance with social skills. No history of trying lower-level routine outpatient therapy.",
            "treatment": "Intensive Outpatient Program (IOP)",
            "insurance_provider": "Cigna",
            "ground_truth": "Likely Denial. The documentation is missing evidence of a failure of lower-level routine outpatient care prior to requesting IOP."
        },
        {
            "diagnosis": "Generalized Anxiety Disorder",
            "symptoms": "Moderate anxiety, poor recovery environment at home, but highly engaged in treatment and no medical comorbidities.",
            "treatment": "Partial Hospitalization Program (PHP)",
            "insurance_provider": "LOCUS",
            "ground_truth": "Likely Denial or Step-Down. Based on LOCUS dimensions, the patient's moderate symptoms and high engagement do not justify the intensity of PHP."
        }
    ]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_dir = os.path.join(project_root, "data", "chroma_db")
    retriever = ClinicalRetriever(persist_directory=persist_dir)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Generating Pipeline Outputs for the 3 Test Cases...")
    for idx, case in enumerate(test_cases, 1):
        print(f"  [{idx}/3] Processing: {case['insurance_provider']}")
        
        question = f"Diagnosis: {case['diagnosis']}, Symptoms: {case['symptoms']}, Treatment: {case['treatment']}, Provider: {case['insurance_provider']}"
        query = f"{case['diagnosis']} {case['symptoms']} {case['treatment']}"
        
        raw_docs = retriever.get_relevant_context(
            query=query,
            insurance_provider=case["insurance_provider"],
            top_k=8
        )
        case_contexts = [doc.page_content for doc in raw_docs]
        
        inputs = {
            "diagnosis": case["diagnosis"],
            "symptoms": case["symptoms"],
            "treatment": case["treatment"],
            "insurance_provider": case["insurance_provider"],
            "additional_notes": "None provided."
        }
        answer = generate_audit_report(inputs)
        
        questions.append(question)
        answers.append(answer)
        contexts.append(case_contexts)
        ground_truths.append(case["ground_truth"])

    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }

    dataset = Dataset.from_dict(data_dict)

    print("\nStarting Ragas Evaluation with LLM Judge (v0.4.3)...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=ragas_llm), 
            ContextPrecision(llm=ragas_llm)
        ]
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
