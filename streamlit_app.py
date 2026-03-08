import streamlit as st
import json
from PIL import Image
from datetime import datetime
from rapidfuzz import process, fuzz
import pytesseract

from med_db import MED_DB
from symptom import symptom_advice
from ocr_utils import extract_text_from_image
from risk_engine import calculate_risk_score
from ollama import Client

# Initialize Ollama client
ollama = Client()

st.set_page_config(page_title="MedSafe AI", layout="wide")

st.title("MedSafe AI Dashboard")
st.write("Welcome to the MedSafe AI interactive front-end dashboard.")

# Basic tab structure
tab1, tab2 = st.tabs(["Upload Prescription", "Analysis"])

with tab1:
    st.header("Upload Prescription Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Prescription", use_container_width=True)
        
        with st.spinner("Extracting text and analyzing..."):
            extracted_text = extract_text_from_image(image)
        st.success("Image processed!")
        
        with st.expander("Show Extracted Text"):
            st.text(extracted_text)

with tab2:
    st.header("Real-Time Analysis")
    st.write("OCR and AI-powered insights will appear here.")
