"""
CSV-based RAG pipeline for agricultural data analysis.
Integrates with existing RAG system to provide crop recommendations.
"""

import os
import pandas as pd
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from .config import settings


class CSVAgriculturalRAG:
    def __init__(self, csv_folder: str = None):
        """
        Initialize CSV-based agricultural RAG system for multiple CSV files.
        """
        # Path to folder containing CSV files
        self.csv_folder = csv_folder or os.path.join(settings.data_docs_dir)
        
        self.vector_db = None
        self.qa_chain = None
        self.dataframes = {}  # dictionary to store multiple DataFrames

    def load_and_prepare_data(self) -> bool:
    """Load all CSV files from data/docs folder and prepare for RAG."""
    try:
        # List all CSVs in the docs folder
        csv_files = [f for f in os.listdir(self.csv_folder) if f.lower().endswith(".csv")]
        if not csv_files:
            print(f"No CSV files found in {self.csv_folder}")
            return False

        docs = []
        for csv_file in csv_files:
            csv_path = os.path.join(self.csv_folder, csv_file)
            df = pd.read_csv(csv_path)
            print(f"Loaded {csv_file} with {len(df)} rows and {len(df.columns)} columns")

            # Turn each row into a LangChain Document
            for idx, row in df.iterrows():
                content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                docs.append(Document(
                    page_content=content,
                    metadata={"row": idx, "file": csv_file}
                ))

        # Split data into chunks for vector embedding
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.split_docs = splitter.split_documents(docs)
        print(f"Created {len(self.split_docs)} document chunks from {len(csv_files)} CSV files")

        return True

    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return False

    
    def build_vector_store(self):
        """Build FAISS vector store from CSV documents."""
        if not self.load_and_prepare_data():
            return False
            
        try:
            # Initialize embeddings
            if settings.model_provider == "openai":
                embeddings = OpenAIEmbeddings()
            else:
                # Use local embeddings as fallback
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                class LocalEmbeddings:
                    def embed_documents(self, texts):
                        return [list(vec) for vec in model.encode(texts, normalize_embeddings=True)]
                    
                    def embed_query(self, text):
                        return list(model.encode([text], normalize_embeddings=True)[0])
                
                embeddings = LocalEmbeddings()
            
            # Build vector store
            self.vector_db = FAISS.from_documents(split_docs, embeddings)
            print("Vector store built successfully")
            
            # Create QA chain
            if settings.model_provider == "openai":
                llm = ChatOpenAI(model=settings.llm_model, temperature=0)
            else:
                # For local models, you might need to adapt this
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            
            return True
            
        except Exception as e:
            print(f"Error building vector store: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the agricultural data.
        
        Args:
            question: Agricultural question to ask
            
        Returns:
            Dictionary with result, sources, and metadata
        """
        if not self.qa_chain:
            return {
                "result": "CSV RAG system not initialized. Please run build_vector_store() first.",
                "sources": [],
                "error": "Not initialized"
            }
        
        try:
            result = self.qa_chain({"query": question})
            return {
                "result": result['result'],
                "sources": [doc.metadata for doc in result["source_documents"]],
                "source_docs": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
            }
        except Exception as e:
            return {
                "result": f"Error processing query: {e}",
                "sources": [],
                "error": str(e)
            }
    
    def get_crop_recommendations(self, soil_type: str, water_availability: str, season: str) -> Dict[str, Any]:
        """
        Get crop recommendations based on farmer context.
        
        Args:
            soil_type: Type of soil (sandy, loam, clay, etc.)
            water_availability: Water availability level
            season: Growing season
            
        Returns:
            Crop recommendations with data
        """
        queries = [
            f"Which crops perform well in {soil_type} soil?",
            f"What crops require {water_availability} water availability?",
            f"Recommend crops suitable for {season} season",
            f"Compare crop yields for {soil_type} soil conditions"
        ]
        
        results = []
        for query in queries:
            result = self.query(query)
            if result.get("result") and not result.get("error"):
                results.append({
                    "query": query,
                    "answer": result["result"],
                    "sources": result["sources"]
                })
        
        return {
            "recommendations": results,
            "context": {
                "soil_type": soil_type,
                "water_availability": water_availability,
                "season": season
            }
        }


# Global instance
csv_rag_instance = None

def get_csv_rag() -> CSVAgriculturalRAG:
    """Get or create global CSV RAG instance."""
    global csv_rag_instance
    if csv_rag_instance is None:
        csv_rag_instance = CSVAgriculturalRAG()
    return csv_rag_instance

def initialize_csv_rag() -> bool:
    """Initialize CSV RAG system."""
    csv_rag = get_csv_rag()
    return csv_rag.build_vector_store()

def query_csv_data(question: str) -> Dict[str, Any]:
    """Query CSV agricultural data."""
    csv_rag = get_csv_rag()
    return csv_rag.query(question)

