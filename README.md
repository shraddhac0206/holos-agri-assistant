## Holos Agri Assistant

End-to-end chatbot that collects farmer context, retrieves agronomy knowledge via RAG, runs Holos CSM presets (X & CUL) for localized recommendations, and responds in text + audio.

### Features
- Collects farmer inputs: location, crop, season, soil, water
- RAG over crop guides/best practices/CSM presets using FAISS or Chroma
- GPT/OpenAI or local LLMs (Llama/Mistral via Ollama) for reasoning
- Mock Holos CSM preset runner that parses X/CUL and applies heuristics
- FastAPI backend; Streamlit UI; gTTS/ElevenLabs TTS
- Handles past, present, future queries with conversation memory

### Quickstart
1. Create virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```
2. Set environment variables:
```bash
copy .env.example .env  # or cp .env.example .env
```
Edit `.env` with keys (OPENAI_API_KEY or OLLAMA_BASE_URL, ELEVENLABS_API_KEY optional).

3. Ingest docs into vector store:
```bash
python -m app.ingest
```

4. Run API:
```bash
uvicorn app.api:app --reload
```

5. Run UI:
```bash
streamlit run app/ui.py
```

### Project Structure
```
app/
  api.py                # FastAPI app
  config.py             # settings
  conversation.py       # stateful conversation manager
  rag.py                # embeddings, retriever, chain
  csm_runner.py         # Holos CSM preset runner (X, CUL)
  tts.py                # gTTS / ElevenLabs
  ingest.py             # load docs into FAISS/Chroma
  utils.py
data/
  docs/                 # sample crop guides
  csm/                  # sample X & CUL presets
stores/
  chroma/ or faiss/     # vector stores
```

### Notes
- Swap vector store between FAISS/Chroma via `RAG_STORE` in `.env`.
- Use `MODEL_PROVIDER`=`openai` or `ollama`. Local models require Ollama installed.


