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
    def __init__(self, csv_path: str = None):
        """
        Initialize CSV-based agricultural RAG system.
        
        Args:
            csv_path: Path to CSV file. If None, looks for KernBlockReportCSV.csv in data folder.
        """
        self.csv_path = csv_path or os.path.join(settings.data_docs_dir, "KernBlockReportCSV.csv")
        self.vector_db = None
        self.qa_chain = None
        self.df = None
        
    def load_and_prepare_data(self) -> bool:
        """Load CSV data and prepare documents for vectorization."""
        try:
            if not os.path.exists(self.csv_path):
                print(f"CSV file not found: {self.csv_path}")
                return False
                
            # Load CSV
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Convert rows into Documents
            docs = []
            for idx, row in self.df.iterrows():
                content = " | ".join([f"{col}: {row[col]}" for col in self.df.columns])
                docs.append(Document(page_content=content, metadata={"row": idx}))
            
            # Split Documents into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n", "|", " ", ""]
            )
            split_docs = splitter.split_documents(docs)
            print(f"Created {len(split_docs)} document chunks")
            
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

