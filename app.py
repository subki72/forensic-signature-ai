import streamlit as st
import requests
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Legal AI - Signature Verifier", layout="centered")

st.title("Legal Document AI")
st.subheader("Forensic Signature Verification System")
st.markdown("---")

# Layout for Image Uploads
col1, col2 = st.columns(2)

with col1:
    st.write("**Master Document (Genuine)**")
    file_asli = st.file_uploader("Upload Master Signature", type=["jpg", "jpeg", "png"], key="master")
    if file_asli:
        st.image(Image.open(file_asli), use_container_width=True)

with col2:
    st.write("**Questioned Document (Test)**")
    file_uji = st.file_uploader("Upload Test Signature", type=["jpg", "jpeg", "png"], key="test")
    if file_uji:
        st.image(Image.open(file_uji), use_container_width=True)

st.markdown("---")

# Execution Button
if st.button("Verify Signature", use_container_width=True):
    if file_asli and file_uji:
        with st.spinner("Analyzing ink stroke anatomy..."):
            try:
                # Prepare files for the FastAPI payload (Keys must match FastAPI parameters)
                files = {
                    "file_asli": (file_asli.name, file_asli.getvalue(), file_asli.type),
                    "file_uji": (file_uji.name, file_uji.getvalue(), file_uji.type)
                }

                # Send request to local API
                response = requests.post("http://localhost:8000/verify", files=files)

                if response.status_code == 200:
                    result = response.json()["verifikasi"]

                    # Display results 
                    # Note: API string outputs are matched directly from main.py
                    if result["status"] == "ASLI (TERVERIFIKASI)":
                        st.success(f"Status: VERIFIED [{result['status']}]")
                    else:
                        st.error(f"Status: REJECTED [{result['status']}]")

                    st.info(f"Similarity Score: {result['skor_kemiripan']} (System Threshold: {result['threshold_sistem']})")
                    st.write(f"**AI Analysis:** {result['hasil_analisa']}")

                else:
                    st.error(f"API Error: {response.text}")
                    
            except Exception as e:
                st.error("Connection failed. Ensure the FastAPI backend (uvicorn) is running.")
    else:
        st.warning("Please upload both signature documents before proceeding.")