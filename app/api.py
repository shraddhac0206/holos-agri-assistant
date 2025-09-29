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


app = FastAPI(title="Holos Agri Assistant")

session = ConversationManager()


class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any] | None = None
    timeframe: str | None = None  # past | present | future


@app.get("/")
def root():
    return {"name": "Holos Agri Assistant", "status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    if req.context:
        session.update_context(req.context)
    if req.timeframe:
        session.update_context({"query_timeframe": req.timeframe})

    session.append_user(req.message)

    missing = session.missing_fields()
    if missing:
        prompt = (
            "I need a bit more info to help you. Please share: " + ", ".join(missing)
        )
        session.append_assistant(prompt)
        return {"reply": prompt, "need_more_info": True, "missing": missing}

    context_str = summarize_context(session.context)
    docs = retrieve(req.message + "\n" + context_str, k=4)
    doc_context = "\n\n".join([d.page_content[:1200] for d in docs])

    csm = run_holos_csm(session.context)

    reply = (
        "Holos Agri Assistant recommendations based on your context and knowledge base:\n\n"
        f"Context: {context_str}\n\n"
        f"Key references:\n{doc_context}\n\n"
        f"CSM insights (score {csm['score']}):\n- " + "\n- ".join(csm["recommendations"]) + "\n\n"
        "This is faster than traditional CSMs, localized, and farmer-friendly."
    )

    session.append_assistant(reply)

    audio_dir = os.path.join("data", "tts")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"reply_{uuid.uuid4().hex}.mp3")
    synthesize_speech(reply, audio_path)

    return {"reply": reply, "audio_path": audio_path, "need_more_info": False}


@app.post("/ingest")
def ingest():
    from .ingest import main as ingest_main

    ingest_main()
    return {"status": "ingested"}


