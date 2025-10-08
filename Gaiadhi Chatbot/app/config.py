import os
from dotenv import load_dotenv

# Load environment variables if any exist (optional)
load_dotenv()


class Settings:
    # Core model configuration
    model_provider: str = os.getenv("MODEL_PROVIDER", "openai").lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # Vector store settings for document retrieval
    rag_store: str = os.getenv("RAG_STORE", "chroma").lower()  # chroma | faiss
    vector_dir: str = os.getenv("VECTOR_DIR", os.path.join(os.getcwd(), "stores"))

    # Models used for embeddings and chatbot responses
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Text-to-speech API (optional)
    elevenlabs_api_key: str | None = os.getenv("ELEVENLABS_API_KEY")

    # Local data directories — no external API used
    data_docs_dir: str = os.path.join(os.getcwd(), "data", "docs")        # text docs
    data_csm_dir: str = os.path.join(os.getcwd(), "data", "csm")          # crop simulation files
    data_weather_dir: str = os.path.join(os.getcwd(), "data", "weather")  # local weather CSV or JSON files
    data_regional_dir: str = os.path.join(os.getcwd(), "data", "regional")# region-level data

settings = Settings()
