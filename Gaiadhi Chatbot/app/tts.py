from __future__ import annotations

import os
from typing import Optional

from gtts import gTTS

from .config import settings


def synthesize_speech(text: str, out_path: str, voice: Optional[str] = None) -> str:
    if settings.elevenlabs_api_key:
        try:
            from elevenlabs import generate, save, set_api_key

            set_api_key(settings.elevenlabs_api_key)
            audio = generate(text=text, voice=voice or "Bella", model="eleven_multilingual_v2")
            save(audio, out_path)
            return out_path
        except Exception:
            pass
    tts = gTTS(text)
    tts.save(out_path)
    return out_path




