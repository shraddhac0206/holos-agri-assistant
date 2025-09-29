from __future__ import annotations

from typing import Dict, Any, List

from .utils import normalize_farmer_context


REQUIRED_FIELDS = ["location", "crop", "season", "soil", "water"]


class ConversationManager:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.context: Dict[str, Any] = {k: "" for k in REQUIRED_FIELDS}
        self.context["query_timeframe"] = "present"

    def append_user(self, text: str):
        self.history.append({"role": "user", "content": text})

    def append_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def update_context(self, partial: Dict[str, Any]):
        self.context.update(normalize_farmer_context(partial))

    def missing_fields(self) -> List[str]:
        return [k for k in REQUIRED_FIELDS if not self.context.get(k)]

    def need_more_info(self) -> bool:
        return len(self.missing_fields()) > 0


