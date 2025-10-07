"""
Multi-source RAG pipeline supporting various file types and data sources.
Handles documents, CSV files, weather data, images, and regional context.
"""

import os
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer

from .config import settings


class MultiSourceRAG:
    def __init__(self):
        """Initialize multi-source RAG system."""
        self.vector_db = None
        self.qa_chain = None
        self.weather_data = {}
        self.regional_context = {}
        
    def get_embedding_model(self):
        """Get embedding model based on configuration."""
        if settings.model_provider == "openai":
            return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
        
        # Local embeddings fallback
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        class LocalEmbeddings:
            def embed_documents(self, texts):
                return [list(vec) for vec in model.encode(texts, normalize_embeddings=True)]
            
            def embed_query(self, text):
                return list(model.encode([text], normalize_embeddings=True)[0])
        
        return LocalEmbeddings()
    
    def process_csv_files(self) -> List[Document]:
        """Process all CSV files in the data directory."""
        docs = []
        csv_dir = settings.data_docs_dir
        
        for filename in os.listdir(csv_dir):
            if filename.lower().endswith('.csv'):
                filepath = os.path.join(csv_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    print(f"Processing CSV: {filename} with {len(df)} rows")
                    
                    # Convert each row to a document
                    for idx, row in df.iterrows():
                        content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                        metadata = {
                            "source": filename,
                            "type": "csv_data",
                            "row": idx,
                            "columns": list(df.columns)
                        }
                        docs.append(Document(page_content=content, metadata=metadata))
                        
                except Exception as e:
                    print(f"Error processing CSV {filename}: {e}")
        
        return docs
    
    def process_weather_files(self) -> List[Document]:
        """Process weather data files."""
        docs = []
        weather_dir = settings.data_weather_dir
        
        if not os.path.exists(weather_dir):
            return docs
            
        for filename in os.listdir(weather_dir):
            filepath = os.path.join(weather_dir, filename)
            try:
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(filepath)
                    # Weather-specific processing
                    for idx, row in df.iterrows():
                        content = f"Weather data for {row.get('region', 'unknown')}: " + \
                                 " | ".join([f"{col}: {row[col]}" for col in df.columns])
                        metadata = {
                            "source": filename,
                            "type": "weather_data",
                            "region": row.get('region', 'unknown'),
                            "date": row.get('date', 'unknown')
                        }
                        docs.append(Document(page_content=content, metadata=metadata))
                        
                elif filename.lower().endswith('.json'):
                    with open(filepath, 'r') as f:
                        weather_data = json.load(f)
                    
                    # Process weather JSON data
                    content = f"Weather patterns and averages: {json.dumps(weather_data, indent=2)}"
                    metadata = {
                        "source": filename,
                        "type": "weather_data",
                        "format": "json"
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
                    
            except Exception as e:
                print(f"Error processing weather file {filename}: {e}")
        
        return docs
    
    def process_regional_files(self) -> List[Document]:
        """Process region-specific context documents."""
        docs = []
        regional_dir = settings.data_regional_dir
        
        if not os.path.exists(regional_dir):
            return docs
            
        for filename in os.listdir(regional_dir):
            filepath = os.path.join(regional_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Extract region from filename or content
                region = filename.replace('.txt', '').replace('.md', '')
                
                metadata = {
                    "source": filename,
                    "type": "regional_context",
                    "region": region
                }
                docs.append(Document(page_content=content, metadata=metadata))
                
            except Exception as e:
                print(f"Error processing regional file {filename}: {e}")
        
        return docs
    
    def process_document_files(self) -> List[Document]:
        """Process text documents (txt, md, pdf)."""
        docs = []
        docs_dir = settings.data_docs_dir
        
        for filename in os.listdir(docs_dir):
            if filename.lower().endswith(('.txt', '.md', '.pdf')):
                filepath = os.path.join(docs_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    metadata = {
                        "source": filename,
                        "type": "document"
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
                    
                except Exception as e:
                    print(f"Error processing document {filename}: {e}")
        
        return docs
    
    def build_comprehensive_vector_store(self):
        """Build vector store from all available data sources."""
        print("Building comprehensive multi-source vector store...")
        
        all_docs = []
        
        # Process different data sources
        print("Processing CSV files...")
        all_docs.extend(self.process_csv_files())
        
        print("Processing weather data...")
        all_docs.extend(self.process_weather_files())
        
        print("Processing regional context...")
        all_docs.extend(self.process_regional_files())
        
        print("Processing documents...")
        all_docs.extend(self.process_document_files())
        
        if not all_docs:
            print("No documents found to process")
            return False
        
        print(f"Total documents processed: {len(all_docs)}")
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n", "|", " ", ""]
        )
        split_docs = splitter.split_documents(all_docs)
        print(f"Created {len(split_docs)} document chunks")
        
        # Build vector store
        embeddings = self.get_embedding_model()
        vector_dir = settings.vector_dir
        os.makedirs(vector_dir, exist_ok=True)
        
        if settings.rag_store == "faiss":
            self.vector_db = FAISS.from_documents(split_docs, embedding=embeddings)
            self.vector_db.save_local(os.path.join(vector_dir, "faiss"))
        else:
            persist_dir = os.path.join(vector_dir, "chroma")
            self.vector_db = Chroma.from_documents(
                split_docs, 
                embedding=embeddings, 
                metadatas=[doc.metadata for doc in split_docs],
                persist_directory=persist_dir
            )
            self.vector_db.persist()
        
        # Create QA chain
        if settings.model_provider == "openai":
            llm = ChatOpenAI(model=settings.llm_model, temperature=0)
        else:
            # For local models, you might need to adapt this
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        print("Multi-source vector store built successfully")
        return True
    
    def query_with_context(self, question: str, farmer_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query with farmer context for personalized responses.
        
        Args:
            question: User question
            farmer_context: Farmer's location, crop, soil, etc.
            
        Returns:
            Enhanced response with context-aware recommendations
        """
        if not self.qa_chain:
            return {
                "result": "RAG system not initialized",
                "sources": [],
                "error": "Not initialized"
            }
        
        # Enhance question with context
        location = farmer_context.get("location", "")
        crop = farmer_context.get("crop", "")
        soil = farmer_context.get("soil", "")
        season = farmer_context.get("season", "")
        
        enhanced_question = f"""
        Context: Location: {location}, Crop: {crop}, Soil: {soil}, Season: {season}
        
        Question: {question}
        
        Please provide region-specific recommendations based on the available data.
        """
        
        try:
            result = self.qa_chain({"query": enhanced_question})
            
            # Categorize sources by type
            source_types = {}
            for doc in result["source_documents"]:
                doc_type = doc.metadata.get("type", "unknown")
                if doc_type not in source_types:
                    source_types[doc_type] = []
                source_types[doc_type].append({
                    "source": doc.metadata.get("source", "unknown"),
                    "region": doc.metadata.get("region", ""),
                    "content": doc.page_content[:200] + "..."
                })
            
            return {
                "result": result['result'],
                "sources": [doc.metadata for doc in result["source_documents"]],
                "source_types": source_types,
                "context_used": farmer_context
            }
            
        except Exception as e:
            return {
                "result": f"Error processing query: {e}",
                "sources": [],
                "error": str(e)
            }
    
    def get_weather_insights(self, location: str, season: str) -> Dict[str, Any]:
        """Get weather insights for specific location and season."""
        # This would integrate with weather APIs or local weather data
        return {
            "location": location,
            "season": season,
            "insights": "Weather data analysis would go here",
            "recommendations": ["Monitor soil moisture", "Adjust irrigation schedule"]
        }


# Global instance
multi_rag_instance = None

def get_multi_rag() -> MultiSourceRAG:
    """Get or create global multi-source RAG instance."""
    global multi_rag_instance
    if multi_rag_instance is None:
        multi_rag_instance = MultiSourceRAG()
    return multi_rag_instance

def initialize_multi_rag() -> bool:
    """Initialize multi-source RAG system."""
    multi_rag = get_multi_rag()
    return multi_rag.build_comprehensive_vector_store()

def query_multi_source(question: str, farmer_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Query multi-source RAG system."""
    multi_rag = get_multi_rag()
    context = farmer_context or {}
    return multi_rag.query_with_context(question, context)

