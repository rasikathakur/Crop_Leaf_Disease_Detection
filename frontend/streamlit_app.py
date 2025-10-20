import streamlit as st
import requests
from PIL import Image
import io
from datetime import datetime
import os


API_URL = os.getenv("API_URL","http://localhost:8000")  

st.set_page_config(
    page_title="Crop Leaf Disease Detection",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🌱 Crop Leaf Disease Detection")
st.markdown(
    """
    <style>
    .main {background-color: #f8fff8;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stFileUploader {background-color: #e8f5e9;}
    </style>
    """,
    unsafe_allow_html=True
)

st.header("1️⃣ Upload a Leaf Image for Disease Detection")

with st.form("upload_form", clear_on_submit=True):
    uploaded_file = st.file_uploader(
        "Choose a leaf image (JPEG/PNG, max 10MB)", 
        type=["jpg", "jpeg", "png"]
    )
    submit = st.form_submit_button("Analyze Image")

if submit and uploaded_file:
    st.info("Uploading and analyzing image...")
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    try:
        response = requests.post(f"{API_URL}/upload_leaf_image", files=files, timeout=60)
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction complete!")
            st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)
            st.markdown(f"**Crop Type:** {result['crop_type']}")
            st.markdown(f"**Disease Status:** :green[{result['disease_status']}]")
            st.markdown(f"**Confidence:** {result['confidence_score']:.2%}")
            st.markdown(f"**Timestamp:** {result['timestamp']}")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

st.divider()
st.header("2️⃣ Prediction History & Statistics")

with st.expander("Show Prediction History", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        limit = st.number_input("Results Limit", min_value=1, max_value=100, value=10)
    with col2:
        crop_type = st.text_input("Filter by Crop Type (optional)")
    with col3:
        disease_status = st.selectbox(
            "Filter by Disease Status (optional)",
            options=["", "Healthy", "Early_Disease", "Severe_Disease"]
        )

    params = {"limit": limit}
    if crop_type:
        params["crop_type"] = crop_type
    if disease_status:
        params["disease_status"] = disease_status

    if st.button("Refresh History"):
        st.session_state["refresh"] = True

    if "refresh" not in st.session_state or st.session_state["refresh"]:
        try:
            response = requests.get(f"{API_URL}/get_results", params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                st.markdown(f"**Total Predictions:** {data['total_predictions']}")
                st.markdown("**Disease Distribution:**")
                st.write(data["disease_distribution"])
                st.markdown("---")
                for r in data["results"]:
                    st.markdown(
                        f"**ID:** {r['leaf_id']} | **Crop:** {r['crop_type']} | "
                        f"**Disease:** :green[{r['disease_status']}] | "
                        f"**Confidence:** {r['confidence_score']:.2%} | "
                        f"**Time:** {r['timestamp']}"
                    )
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
        st.session_state["refresh"] = False

st.divider()

st.header("3️⃣ Model Performance")
try:
    st.image("training_history.png", caption="Model Accuracy and Loss During Training", use_column_width=True)
except Exception as e:
    st.warning("Training history plot not found. Place training_history.png in the frontend directory to view model training metrics.")

st.divider()
