from __future__ import annotations

from .config import settings
from .rag import build_vector_store_from_files


def main():
    build_vector_store_from_files(settings.data_docs_dir, store=settings.rag_store)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()




