from typing import TypedDict


class PlanExecute(TypedDict):
    input: str
    chat_history: list
    response: str
