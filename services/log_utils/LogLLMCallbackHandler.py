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
                content = (m["content"] or "").replace("\n", " ").strip()
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    args_raw = fn.get("arguments", "")
                    try:
                        args_str = json.dumps(json.loads(args_raw), ensure_ascii=False)
                    except Exception:
                        args_str = args_raw
                    content += f" [tool_call → {fn.get('name', '?')}({args_str[:400]})]"
                if len(content) > 5000:
                    content = content[:5000] + "..."
                lines.append(f"  [{m['type'].upper()}]: {content}")
        return "\n".join(lines)

    def on_chat_model_start(self, serialized, _messages, **kwargs):
        self.call_count += 1
        if not self._enabled:
            return
        llm_kwargs = serialized.get("kwargs", {})
        model = llm_kwargs.get("model_name", "unknown")
        temperature = llm_kwargs.get("temperature", "?")

        def _msg_dict(m):
            d = {"type": m.type, "content": m.content}
            tc = getattr(m, "tool_calls", None) or m.additional_kwargs.get("tool_calls", [])
            if tc:
                d["tool_calls"] = tc
            return d

        msgs = [[_msg_dict(m) for m in grp] for grp in _messages]
        self._log_entries.append({"call": self.call_count, "model": model, "messages": msgs})
        formatted = self._format_messages(msgs)
        log_message(
            step_name=f"LLM API call #{self.call_count} model={model} temperature={temperature}",
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
        tool_calls = []
        if hasattr(gen, "message"):
            tool_calls = gen.message.additional_kwargs.get("tool_calls", [])
        if text:
            if self._log_entries:
                self._log_entries[-1]["response"] = text
            log_message(step_name=f"LLM response #{self.call_count}", color="Magenta", messages=text.splitlines())
        for tc in tool_calls:
            fn = tc.get("function", {})
            args_raw = fn.get("arguments", "")
            try:
                args_str = json.dumps(json.loads(args_raw), ensure_ascii=False)
            except Exception:
                args_str = args_raw
            log_message(
                step_name=f"LLM tool_call #{self.call_count}",
                color="Magenta",
                messages=[f"{fn.get('name', '?')}: {args_str[:500]}"],
            )
