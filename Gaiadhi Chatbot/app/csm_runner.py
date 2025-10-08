from __future__ import annotations

import json
import os
from typing import Dict, Any

from .config import settings


def load_csm_presets() -> Dict[str, Any]:
    # This function loads all crop simulation model (CSM) preset files
    # from the "data/csm" folder defined in config.py.
    # Presets can be in .x, .cul, or .json formats.

    presets: Dict[str, Any] = {}
    for root, _, files in os.walk(settings.data_csm_dir):
        for fn in files:
            if fn.lower().endswith((".x", ".cul", ".json")):
                path = os.path.join(root, fn)
                try:
                    # Read the file’s content safely
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    # If a file can’t be opened, skip it
                    continue
                # Parse and store the file content
                presets[fn] = parse_csm_content(content, fn)
    return presets


def parse_csm_content(content: str, name: str) -> Dict[str, Any]:
    # This helper function tries to understand the preset file.
    # If the file is JSON, it loads it as structured data.
    # If it's not JSON (like .x or .cul), it just stores the raw text.
    try:
        return json.loads(content)
    except Exception:
        return {"name": name, "raw": content}


def run_holos_csm(context: Dict[str, Any]) -> Dict[str, Any]:
    # This is the main function that simulates basic crop recommendations
    # based on soil and water conditions provided by the user.

    # Load any CSM preset files from the data folder
    presets = load_csm_presets()

    # Get user context values
    crop = (context.get("crop") or "").lower()
    soil = (context.get("soil") or "").lower()
    water = (context.get("water") or "").lower()

    # Start a "score" that represents how favorable the conditions are
    score = 0

    # Adjust score based on soil type
    if "loam" in soil:
        score += 2
    if "clay" in soil:
        score += 1
    if "sandy" in soil:
        score -= 1

    # Adjust score based on water availability
    if "low" in water or "scarce" in water:
        score -= 1
    if "medium" in water:
        score += 0
    if "high" in water or "canal" in water:
        score += 1

    # Basic farming recommendations, always included
    recommendations = [
        "Use soil moisture checks before irrigation.",
        "Apply region-specific fertilizer guidelines.",
        "Prefer integrated pest management; monitor weekly.",
    ]

    # Add a crop-specific message if the crop is known
    if crop:
        recommendations.insert(0, f"Follow best practices for {crop} in your region.")

    # Add water-specific advice depending on score
    if score <= 0:
        recommendations.append("Consider drought-tolerant varieties or mulching to retain moisture.")
    else:
        recommendations.append("Irrigate on a 3–5 day cadence depending on evapotranspiration.")

    # Return all results for chatbot use
    return {
        "score": score,                        # How favorable the conditions are
        "presets_used": list(presets.keys())[:5],  # Which CSM preset files were found (up to 5)
        "recommendations": recommendations,     # Final farming recommendations list
    }



