from __future__ import annotations
from typing import Dict, Any, List

from .utils import normalize_farmer_context

# These are the key pieces of information the chatbot needs from every user
# Example: location (Punjab), crop (rice), season (Kharif), soil (loamy), water (moderate)
REQUIRED_FIELDS = ["location", "crop", "season", "soil", "water"]


class ConversationManager:
    """
    This class manages the conversation between the farmer (user) and the chatbot.
    It keeps track of what has been said and what information is still missing.
    """

    def __init__(self):
        # History saves the full chat — both user and assistant messages
        self.history: List[Dict[str, str]] = []

        # Context stores key details about the farmer's situation (location, crop, etc.)
        # All are initially empty strings
        self.context: Dict[str, Any] = {k: "" for k in REQUIRED_FIELDS}

        # By default, we assume the question is about the "present" timeframe
        self.context["query_timeframe"] = "present"

    def append_user(self, text: str):
        # Adds the user's message to the chat history
        self.history.append({"role": "user", "content": text})

    def append_assistant(self, text: str):
        # Adds the chatbot's reply to the chat history
        self.history.append({"role": "assistant", "content": text})

    def update_context(self, partial: Dict[str, Any]):
        # Updates only specific pieces of context (like crop or soil)
        # normalize_farmer_context() cleans and standardizes user input
        self.context.update(normalize_farmer_context(partial))

    def missing_fields(self) -> List[str]:
        # Checks which key fields (location, crop, etc.) are still missing
        # Returns a list like ["soil", "water"] if not provided yet
        return [k for k in REQUIRED_FIELDS if not self.context.get(k)]

    def need_more_info(self) -> bool:
        # Returns True if any required field is missing
        # This helps the chatbot decide if it should ask for more details
        return len(self.missing_fields()) > 0

