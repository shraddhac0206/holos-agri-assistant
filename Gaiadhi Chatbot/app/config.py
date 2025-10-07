import os
from dotenv import load_dotenv


load_dotenv()


class Settings:
    model_provider: str = os.getenv("MODEL_PROVIDER", "openai").lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    rag_store: str = os.getenv("RAG_STORE", "chroma").lower()  # chroma | faiss
    vector_dir: str = os.getenv("VECTOR_DIR", os.path.join(os.getcwd(), "stores"))

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    elevenlabs_api_key: str | None = os.getenv("ELEVENLABS_API_KEY")

    weather_api_url: str | None = os.getenv("WEATHER_API_URL")
    weather_api_key: str | None = os.getenv("WEATHER_API_KEY")

    data_docs_dir: str = os.path.join(os.getcwd(), "data", "docs")
    data_csm_dir: str = os.path.join(os.getcwd(), "data", "csm")
    data_weather_dir: str = os.path.join(os.getcwd(), "data", "weather")
    data_images_dir: str = os.path.join(os.getcwd(), "data", "images")
    data_regional_dir: str = os.path.join(os.getcwd(), "data", "regional")


settings = Settings()



