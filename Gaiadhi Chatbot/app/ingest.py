from __future__ import annotations

# Import project settings (paths, storage type, etc.)
from .config import settings

# Import the function that actually builds the vector database
from .rag import build_vector_store_from_files


def main():
    """
    Main ingestion function.
    Purpose:
        - Reads all files (e.g., CSV, TXT, PDF) from the data/docs folder.
        - Converts them into embeddings using your chosen embedding model.
        - Stores them in the selected vector store (FAISS or Chroma).
    Why it's needed:
        - This step lets the chatbot perform RAG (Retrieval-Augmented Generation),
          so it can answer user questions using local documents instead of only the LLM.
    """
    
    # Build vector store (embeddings) from all files in data/docs directory
    build_vector_store_from_files(settings.data_docs_dir, store=settings.rag_store)

    # Print confirmation once indexing is complete
    print("Ingestion complete — all documents indexed successfully.")


# Allow this script to be run directly from the command line
if __name__ == "__main__":
    main()



