from __future__ import annotations

import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


st.set_page_config(page_title="Holos Agri Assistant", page_icon="🌾")
st.title("Holos Agri Assistant ")

with st.sidebar:
    st.header("Farmer Context")
    location = st.text_input("Location", "")
    crop = st.text_input("Crop", "")
    season = st.text_input("Season", "")
    soil = st.text_input("Soil", "")
    water = st.text_input("Water Availability", "")
    timeframe = st.selectbox("Timeframe", ["past", "present", "future"], index=1)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Ingest All Data Sources"):
            with st.spinner("Ingesting all data sources (docs, CSV, weather, regional)..."):
                r = requests.post(f"{API_BASE}/ingest", timeout=180)
                if r.ok:
                    result = r.json()
                    st.success(f"Ingested: {result.get('status')}")
                    
                    if result.get('multi_source_rag'):
                        st.success(" Multi-source RAG system loaded!")
                    elif result.get('multi_source_error'):
                        st.warning(f" Multi-source RAG: {result['multi_source_error']}")
                    
                    if result.get('csv_rag'):
                        st.success(" CSV agricultural data loaded!")
                    elif result.get('csv_error'):
                        st.warning(f" CSV data: {result['csv_error']}")
                else:
                    st.error(f"Error: {r.status_code} {r.text}")
    
    with col2:
        if st.button("Check Data Sources"):
            r = requests.get(f"{API_BASE}/data-sources")
            if r.ok:
                sources = r.json()
                st.write("**Available Data Sources:**")
                for source, available in sources.items():
                    status = "✅" if available else "❌"
                    st.write(f"{status} {source.title()}")
            else:
                st.error(f"Error checking sources: {r.status_code}")

chat_area = st.container()
# Enhanced chat interface
st.subheader(" Chat with Holos Agri Assistant")

# Sample questions
sample_questions = [
    "Should I irrigate today?",
    "What crops perform best in sandy soil?",
    "Compare wheat and maize water requirements",
    "What's the weather forecast for next week?",
    "Recommend fertilizers for rice cultivation",
    "Predict rice yield for my field",
    "Analyze LAI trends for Punjab region",
    "Compare rice varieties IR64 vs Pusa Basmati",
    "What's my water use efficiency?",
    "How can I improve nitrogen efficiency?"
]

selected_question = st.selectbox("Choose a sample question or type your own:", ["Custom question..."] + sample_questions)

if selected_question == "Custom question...":
    user_msg = st.text_input("Your question", "")
else:
    user_msg = selected_question

if st.button("Send", type="primary"):
    payload = {
        "message": user_msg,
        "context": {
            "location": location,
            "crop": crop,
            "season": season,
            "soil": soil,
            "water": water,
        },
        "timeframe": timeframe,
    }
    resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
    if resp.ok:
        data = resp.json()
        chat_area.write(data.get("reply", ""))
        audio_path = data.get("audio_path")
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
    else:
        st.error(f"Error: {resp.status_code} {resp.text}")



