from typing import Dict, Any


def normalize_farmer_context(raw: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {
        "location": (raw.get("location") or "").strip(),
        "crop": (raw.get("crop") or "").strip(),
        "season": (raw.get("season") or "").strip(),
        "soil": (raw.get("soil") or "").strip(),
        "water": (raw.get("water") or "").strip(),
        "query_timeframe": (raw.get("query_timeframe") or "present").strip().lower(),
    }
    return cleaned


def summarize_context(context: Dict[str, Any]) -> str:
    bits = []
    for key in ["location", "crop", "season", "soil", "water", "query_timeframe"]:
        val = context.get(key)
        if val:
            bits.append(f"{key}: {val}")
    return ", ".join(bits)




