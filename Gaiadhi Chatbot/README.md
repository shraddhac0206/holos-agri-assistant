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

5. Run Chat UI:
```bash
streamlit run app/ui.py
```

6. Run Dashboard:
```bash
streamlit run app/dashboard.py
```

### Project Structure
```
app/
  api.py                # FastAPI app
  config.py             # settings
  conversation.py       # stateful conversation manager
  rag.py                # embeddings, retriever, chain
  csv_rag.py            # CSV-based agricultural data RAG
  multi_source_rag.py   # Multi-source RAG (docs, CSV, weather, regional)
  lai_analyzer.py       # Leaf Area Index analysis (Sentinel satellite data)
  yield_predictor.py    # AI yield prediction with water/nitrogen efficiency
  csm_runner.py         # Holos CSM preset runner (X, CUL)
  tts.py                # gTTS / ElevenLabs
  ingest.py             # load docs into FAISS/Chroma
  dashboard.py          # User-friendly CSM dashboard
  ui.py                 # Streamlit chat interface
  utils.py
data/
  docs/                 # crop guides, CSV files (KernBlockReportCSV.csv)
  weather/              # weather data files (CSV, JSON)
  regional/             # region-specific context files
  images/               # agricultural images and graphs
  csm/                  # sample X & CUL presets
  lai/                  # LAI analysis exports
stores/
  chroma/ or faiss/     # vector stores
```

### Notes
- Swap vector store between FAISS/Chroma via `RAG_STORE` in `.env`.
- Use `MODEL_PROVIDER`=`openai` or `ollama`. Local models require Ollama installed.



