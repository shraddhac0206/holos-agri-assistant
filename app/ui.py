from __future__ import annotations

import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


st.set_page_config(page_title="Holos Agri Assistant", page_icon="🌾")
st.title("Holos Agri Assistant 🌾")

with st.sidebar:
    st.header("Farmer Context")
    location = st.text_input("Location", "")
    crop = st.text_input("Crop", "")
    season = st.text_input("Season", "")
    soil = st.text_input("Soil", "")
    water = st.text_input("Water Availability", "")
    timeframe = st.selectbox("Timeframe", ["past", "present", "future"], index=1)
    if st.button("Ingest Docs"):
        r = requests.post(f"{API_BASE}/ingest", timeout=60)
        st.success(f"Ingested: {r.json()}")

chat_area = st.container()
user_msg = st.text_input("Your question", "Should I irrigate today?")
if st.button("Send"):
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


