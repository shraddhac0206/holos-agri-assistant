from __future__ import annotations

import os
import uuid
from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from .config import settings
from .conversation import ConversationManager
from .rag import retrieve
from .tts import synthesize_speech
from .csm_runner import run_holos_csm
from .utils import summarize_context
from .csv_rag import query_csv_data, initialize_csv_rag
from .multi_source_rag import query_multi_source, initialize_multi_rag
from .lai_analyzer import analyze_region_lai
from .yield_predictor import predict_crop_yield


app = FastAPI(title="Holos Agri Assistant") # Creates main FastAPI app for chatbot backend

session = ConversationManager()  # Manages chat history and user context


class ChatRequest(BaseModel): # Defines input data model for /chat endpoint
    message: str
    context: Dict[str, Any] | None = None
    timeframe: str | None = None  # past | present | future


@app.get("/")
def root():
    return {"name": "Holos Agri Assistant", "status": "ok"}  # Health check endpoint


@app.post("/chat")
def chat(req: ChatRequest):
    """Main chatbot endpoint — handles user messages, context, RAG, and response synthesis"""
    # Update conversation context if provided
    if req.context:
        session.update_context(req.context)
    if req.timeframe:
        session.update_context({"query_timeframe": req.timeframe})

    session.append_user(req.message) # Adds user message to conversation memory
    
    # Check if chatbot needs more details (missing fields)
    missing = session.missing_fields()
    if missing:
        prompt = (
            "I need a bit more info to help you. Please share: " + ", ".join(missing)
        )
        session.append_assistant(prompt)
        return {"reply": prompt, "need_more_info": True, "missing": missing}

    context_str = summarize_context(session.context) # Summarize current conversation context
    
    # Multi-source retrieval (weather + regional + CSV + docs)
    multi_results = query_multi_source(req.message, session.context)
    
    # Get traditional RAG results as fallback
    docs = retrieve(req.message + "\n" + context_str, k=4)
    doc_context = "\n\n".join([d.page_content[:1200] for d in docs])
    
    # Get CSV-based agricultural data insights
    csv_results = query_csv_data(req.message)
    csv_insights = ""
    if csv_results.get("result") and not csv_results.get("error"):
        csv_insights = f"\n\nAgricultural Data Analysis:\n{csv_results['result']}"
        if csv_results.get("source_docs"):
            csv_insights += f"\n\nSupporting data: {csv_results['source_docs'][0]}"

    # Run Holos Crop Simulation Model (CSM) logic
    csm = run_holos_csm(session.context)

    # Combine all analysis results into a single response
    reply_parts = [
        " Holos Agri Assistant - Comprehensive Recommendations",
        f"\n Context: {context_str}",
    ]
    
    # Add insights from multi-source RAG
    if multi_results.get("result") and not multi_results.get("error"):
        reply_parts.append(f"\n Multi-Source Analysis:\n{multi_results['result']}")
        
        # Add source breakdown
        source_types = multi_results.get("source_types", {})
        if source_types:
            reply_parts.append("\n Data Sources Used:")
            for source_type, sources in source_types.items():
                reply_parts.append(f"  • {source_type}: {len(sources)} sources")
    
    # Add knowledge base reference summaries
    if doc_context:
        reply_parts.append(f"\n Knowledge Base References:\n{doc_context}")
    
    # Add CSV insights
    if csv_insights:
        reply_parts.append(csv_insights)
    
    # Add CSM-based agronomic recommendations
    reply_parts.extend([
        f"\n CSM Insights (score {csm['score']}):",
        "• " + "\n• ".join(csm["recommendations"])
    ])
    
    reply_parts.extend([
        "\n\n Holos Advantage: Faster than traditional CSMs, localized, scalable, and farmer-friendly.",
        "\n Tip: Ask about specific crops, weather patterns, or regional best practices for detailed insights."
    ])

    reply = "\n".join(reply_parts)

    session.append_assistant(reply)

    audio_dir = os.path.join("data", "tts")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"reply_{uuid.uuid4().hex}.mp3")
    synthesize_speech(reply, audio_path)

    return {"reply": reply, "audio_path": audio_path, "need_more_info": False}


@app.post("/ingest")
def ingest():
    from .ingest import main as ingest_main

    # Ingest traditional documents
    ingest_main()
    
    results = {"status": "ingested"}
    
    # Initialize advanced multi-source RAG (text + structured data)
    try:
        multi_success = initialize_multi_rag()
        results["multi_source_rag"] = multi_success
    except Exception as e:
        results["multi_source_rag"] = False
        results["multi_source_error"] = str(e)
    
    # Initialize CSV RAG as backup
    try:
        csv_success = initialize_csv_rag()
        results["csv_rag"] = csv_success
    except Exception as e:
        results["csv_rag"] = False
        results["csv_error"] = str(e)
    
    return results

@app.post("/csv-query")
def csv_query(req: ChatRequest):
    """Direct query to CSV agricultural data."""
    result = query_csv_data(req.message)
    return result

@app.post("/multi-source-query")
def multi_source_query(req: ChatRequest):
    """Direct query to multi-source RAG system."""
    context = req.context or {}
    result = query_multi_source(req.message, context)
    return result

@app.get("/data-sources")
def get_data_sources():
    """Query across multiple data sources (CSV + documents + weather + region)"""
    sources = {
        "docs": os.path.exists(settings.data_docs_dir) and len(os.listdir(settings.data_docs_dir)) > 0,
        "weather": os.path.exists(settings.data_weather_dir) and len(os.listdir(settings.data_weather_dir)) > 0,
        "regional": os.path.exists(settings.data_regional_dir) and len(os.listdir(settings.data_regional_dir)) > 0,
        "images": os.path.exists(settings.data_images_dir) and len(os.listdir(settings.data_images_dir)) > 0,
        "csm": os.path.exists(settings.data_csm_dir) and len(os.listdir(settings.data_csm_dir)) > 0,
    }
    return sources

@app.post("/predict-yield")
def predict_yield_endpoint(request: dict):
    """Predict crop yield with efficiency analysis."""
    crop_type = request.get("crop_type", "rice")
    parameters = request.get("parameters", {})
    
    result = predict_crop_yield(crop_type, parameters)
    return result

@app.post("/analyze-lai")
def analyze_lai_endpoint(request: dict):
    """Analyze Leaf Area Index for a region."""
    region = request.get("region", "Punjab")
    days_back = request.get("days_back", 365)
    
    result = analyze_region_lai(region, days_back)
    return result

@app.post("/compare-varieties")
def compare_varieties_endpoint(request: dict):
    """Compare yield predictions across crop varieties."""
    from .yield_predictor import get_yield_predictor
    
    crop_type = request.get("crop_type", "rice")
    varieties = request.get("varieties", ["IR64", "Pusa Basmati"])
    parameters = request.get("parameters", {})
    
    predictor = get_yield_predictor()
    result = predictor.compare_varieties(crop_type, varieties, parameters)
    return result



