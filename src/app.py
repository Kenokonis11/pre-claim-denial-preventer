import os
import sys

import streamlit as st

# Ensure src is in the python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.generation import generate_audit_report, get_cached_llm, get_cached_retriever


@st.cache_resource
def load_backend_resources():
    persist_dir = os.path.join(project_root, "data", "chroma_db")
    return get_cached_retriever(persist_dir), get_cached_llm()


st.set_page_config(
    page_title="Clinical Denial Preventer",
    page_icon="Medical",
    layout="wide",
)

st.title("Clinical Denial Preventer")
st.markdown(
    """
This tool evaluates clinical documentation against insurance medical necessity criteria
using a Retrieval-Augmented Generation (RAG) pipeline. It helps validate medical claims
before submission to predict and prevent likely denials.
"""
)

with st.form("clinical_audit_form"):
    st.subheader("Patient Clinical Data")

    col1, col2 = st.columns(2)

    with col1:
        diagnosis = st.text_input("Diagnosis")
        treatment = st.text_input("Requested Treatment")
        insurance_provider = st.selectbox(
            "Insurance Provider",
            options=["HUSKY", "Anthem", "Cigna", "Health New England", "Other"],
        )

    with col2:
        symptoms = st.text_area("Symptoms & Clinical Presentation", height=130)
        additional_notes = st.text_area("Additional Notes (Optional)", height=85)

    submit_button = st.form_submit_button("Generate Audit Report")

if submit_button:
    if not diagnosis or not symptoms or not treatment:
        st.error("Please fill in the required fields: Diagnosis, Symptoms, and Requested Treatment.")
    else:
        with st.spinner("Analyzing clinical criteria against policies..."):
            backend_provider = "LOCUS" if insurance_provider == "Other" else insurance_provider

            inputs = {
                "diagnosis": diagnosis,
                "symptoms": symptoms,
                "treatment": treatment,
                "insurance_provider": backend_provider,
                "additional_notes": additional_notes if additional_notes else "None provided.",
            }

            try:
                retriever, llm = load_backend_resources()
                raw_text = generate_audit_report(inputs, retriever=retriever, llm=llm)

                if "[STATUS: APPROVAL]" in raw_text.upper():
                    st.success("Analysis Complete - Positive Outlook")
                    clean_report = raw_text.replace("[STATUS: APPROVAL]", "", 1).strip()
                elif "[STATUS: DENIAL]" in raw_text.upper():
                    st.error("Analysis Complete - High Risk of Denial / Criteria Not Met")
                    clean_report = raw_text.replace("[STATUS: DENIAL]", "", 1).strip()
                elif "[STATUS: WARNING]" in raw_text.upper():
                    st.warning("Analysis Complete - Insufficient Context or Payer Criteria Missing")
                    clean_report = raw_text.replace("[STATUS: WARNING]", "", 1).strip()
                else:
                    st.info("Analysis Complete - Review Audit Summary Below")
                    clean_report = raw_text

                if "Data Quality Warning:" in clean_report:
                    st.warning(clean_report)

                st.subheader("Audit Results")
                st.markdown(clean_report)

            except Exception as e:
                st.error(f"An error occurred during generating the report: {str(e)}")
                st.info("Please ensure your API keys in the `.env` file are configured correctly.")
