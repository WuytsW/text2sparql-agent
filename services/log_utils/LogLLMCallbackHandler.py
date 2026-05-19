from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from services.log_utils.log import log_message
import os
import json


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

    def _format_messages(self, msgs):
        lines = []
        for grp in msgs:
            for m in grp:
                content = m["content"].replace("\n", " ").strip()
                if len(content) > 5000:
                    content = content[:5000] + "..."
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
        log_message(
            step_name=f"LLM API call #{self.call_count} model={model}",
            color="Blue",
            messages=formatted.splitlines(),
        )

    def on_llm_error(self, error, **kwargs):
        log_message(
            step_name=f"LLM API error #{self.call_count}",
            color="Red",
            messages=[str(error)],
        )

    def on_llm_end(self, response, **kwargs):
        if not self._enabled:
            return
        gen = response.generations[0][0]
        text = gen.text or (gen.message.content if hasattr(gen, "message") else "")
        if text:
            if self._log_entries:
                self._log_entries[-1]["response"] = text
            log_message(
                step_name=f"LLM response #{self.call_count}",
                color="Magenta",
                messages=text.splitlines(),
            )
