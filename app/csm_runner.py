from __future__ import annotations

import json
import os
from typing import Dict, Any

from .config import settings


def load_csm_presets() -> Dict[str, Any]:
    presets: Dict[str, Any] = {}
    for root, _, files in os.walk(settings.data_csm_dir):
        for fn in files:
            if fn.lower().endswith((".x", ".cul", ".json")):
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue
                presets[fn] = parse_csm_content(content, fn)
    return presets


def parse_csm_content(content: str, name: str) -> Dict[str, Any]:
    # Very lightweight parser: supports JSON presets; for .x/.cul we return text blob
    try:
        return json.loads(content)
    except Exception:
        return {"name": name, "raw": content}


def run_holos_csm(context: Dict[str, Any]) -> Dict[str, Any]:
    presets = load_csm_presets()
    crop = (context.get("crop") or "").lower()
    soil = (context.get("soil") or "").lower()
    water = (context.get("water") or "").lower()

    score = 0
    if "loam" in soil:
        score += 2
    if "clay" in soil:
        score += 1
    if "sandy" in soil:
        score -= 1
    if "low" in water or "scarce" in water:
        score -= 1
    if "medium" in water:
        score += 0
    if "high" in water or "canal" in water:
        score += 1

    recommendations = [
        "Use soil moisture checks before irrigation.",
        "Apply region-specific fertilizer guidelines.",
        "Prefer integrated pest management; monitor weekly.",
    ]
    if crop:
        recommendations.insert(0, f"Follow best practices for {crop} in your region.")
    if score <= 0:
        recommendations.append("Consider drought-tolerant varieties or mulching to retain moisture.")
    else:
        recommendations.append("Irrigate on a 3–5 day cadence depending on evapotranspiration.")

    return {
        "score": score,
        "presets_used": list(presets.keys())[:5],
        "recommendations": recommendations,
    }


