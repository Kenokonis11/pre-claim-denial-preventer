import streamlit as st
import os
import sys

# Ensure src is in the python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.generation import generate_audit_report

# 1. Page Configuration & Header
st.set_page_config(
    page_title="Clinical Denial Preventer",
    page_icon="⚕️",
    layout="wide"
)

st.title("Clinical Denial Preventer")
st.markdown("""
This tool evaluates clinical documentation against insurance medical necessity criteria 
using a Retrieval-Augmented Generation (RAG) pipeline. It helps validate medical claims 
before submission to predict and prevent likely denials.
""")

# 2. The Input Form
with st.form("clinical_audit_form"):
    st.subheader("Patient Clinical Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        diagnosis = st.text_input("Diagnosis")
        treatment = st.text_input("Requested Treatment")
        insurance_provider = st.selectbox(
            "Insurance Provider",
            options=["HUSKY", "Anthem", "Cigna", "Health New England", "Other"]
        )
        
    with col2:
        symptoms = st.text_area("Symptoms & Clinical Presentation", height=130)
        additional_notes = st.text_area("Additional Notes (Optional)", height=85)

    # Submit button for the form
    submit_button = st.form_submit_button("Generate Audit Report")

# 3. Form Submission Logic
if submit_button:
    if not diagnosis or not symptoms or not treatment:
        st.error("Please fill in the required fields: Diagnosis, Symptoms, and Requested Treatment.")
    else:
        with st.spinner("Analyzing clinical criteria against policies..."):
            # Map the UI dropdown value to the backend metadata tag
            backend_provider = "LOCUS" if insurance_provider == "Other" else insurance_provider
            
            inputs = {
                "diagnosis": diagnosis,
                "symptoms": symptoms,
                "treatment": treatment,
                "insurance_provider": backend_provider,
                "additional_notes": additional_notes if additional_notes else "None provided."
            }
            
            try:
                # Call the backend function
                report = generate_audit_report(inputs)
                
                # 4. Displaying the Output
                st.subheader("Audit Results")
                
                # Simple heuristic to determine styling based on report content
                if "likely approval" in report.lower() or "meets criteria" in report.lower() and "unlikely" not in report.lower() and "denial" not in report.lower():
                    st.success("Analysis Complete - Positive Outlook")
                elif "denial" in report.lower() or "unlikely" in report.lower() or "missing" in report.lower():
                    st.warning("Analysis Complete - Potential Issues Detected")
                else:
                    st.info("Analysis Complete")
                    
                # Render the markdown report
                st.markdown(report)
                
            except Exception as e:
                st.error(f"An error occurred during generating the report: {str(e)}")
                st.info("Please ensure your API keys in the `.env` file are configured correctly.")
