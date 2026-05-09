import logging
import os
import re
from datetime import datetime

BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

RESET   = "\033[0m"

COLOR_MAP = {
    "Black": BLACK, "Red": RED, "Green": GREEN, "Yellow": YELLOW,
    "Blue": BLUE, "Magenta": MAGENTA, "Cyan": CYAN, "White": WHITE,
}

_THESIS_LOG_DIR = r"G:\Thesis\logs"
_question_log_file: str | None = None
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def set_question_log(question: str):
    global _question_log_file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_question = re.sub(r'[\\/:*"<>|]', "_", question)[:80]
    safe_question = re.sub(r'[?]', "", safe_question)[:80]
    filename = f"[{timestamp}]_{safe_question}.txt"
    os.makedirs(_THESIS_LOG_DIR, exist_ok=True)
    _question_log_file = os.path.join(_THESIS_LOG_DIR, filename)


def _append_to_question_log(text: str):
    if _question_log_file:
        with open(_question_log_file, "a", encoding="utf-8") as f:
            f.write(_strip_ansi(text) + "\n")


def log_message(step_name: str, color: str = "White", messages: list = None):
    colorCode = COLOR_MAP.get(color, WHITE)
    header = f"{colorCode}[{datetime.now().strftime('%H:%M:%S')}][{step_name}]{RESET}"
    logging.info(header)
    _append_to_question_log(header)
    if messages:
        for message in messages:
            line = f"{colorCode}{message}{RESET}"
            print(line)
            _append_to_question_log(line)

def log_warning(step_name: str, error_message: str):
    log_message(step_name, color="Red", messages=[f"Error: {error_message}"])