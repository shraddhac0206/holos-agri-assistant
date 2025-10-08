import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Holos Agri Assistant", layout="centered")

st.title("🌾Holos Agri Assistant")
st.markdown("Ask questions about crops, irrigation, soil, or yield below ")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(f"**You:** {msg['content']}")
    else:
        st.chat_message("assistant").markdown(f"**Holos Assistant:** {msg['content']}")

# Input box
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    try:
        response = requests.post(f"{API_BASE}/chat", json={"message": user_input})
        if response.ok:
            reply = response.json().get("reply", "No reply received.")
        else:
            reply = f"Error: {response.status_code}"
    except Exception as e:
        reply = f"Error connecting to chatbot: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.experimental_rerun()
