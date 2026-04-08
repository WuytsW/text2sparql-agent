import logging
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
import os
import json


BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

RESET   = "\033[0m"

class LogLLMCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self._log_entries = []

    def reset(self, question: str, enabled: bool = True):
        self.call_count = 0
        self._log_entries = []
        self._question = question
        self._start_time = datetime.now().isoformat()
        self._enabled = enabled

    def _flush_to_file(self, sparql: str, log_path: str = "logs/llm_calls.json"):
        if not self._enabled:
            return
        record = {
            "time": self._start_time,
            "question": self._question,
            "total_llm_calls": self.call_count,
            "sparql": sparql,
            "calls": self._log_entries,
        }
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        existing = []
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        existing.append(record)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def _format_messages(self, msgs):
        lines = []
        for grp in msgs:
            for m in grp:
                content = m["content"].replace("\n", " ").strip()
                if len(content) > 1000:
                    content = content[:1000] + "..."
                lines.append(f"  [{m['type'].upper()}]: {content}")
        return "\n".join(lines)

    def on_chat_model_start(self, serialized, _messages, **kwargs):
        self.call_count += 1
        if not self._enabled:
            return
        model = serialized.get("kwargs", {}).get("model_name", "unknown")
        msgs = [[{"type": m.type, "content": m.content} for m in grp] for grp in _messages]
        self._log_entries.append({"call": self.call_count, "model": model, "messages": msgs})
        formatted = self._format_messages(msgs)
        logging.info(f"{BLUE}[LLM API call #{self.call_count}] model={model}\n{formatted}{RESET}")

    def on_llm_end(self, response, **kwargs):
        if not self._enabled:
            return
        gen = response.generations[0][0]
        text = gen.text or (gen.message.content if hasattr(gen, "message") else "")
        if text:
            if self._log_entries:
                self._log_entries[-1]["response"] = text
            logging.info(f"{GREEN}[LLM response #{self.call_count}]:\n{text}{RESET}")