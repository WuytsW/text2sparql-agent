import logging
import os
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

_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "messages.log")

def _append_to_log(text: str):
    os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_message(step_name: str, color: str = "White", messages: list = None):
    colorCode = COLOR_MAP.get(color, WHITE)
    header = f"{colorCode}[{datetime.now().strftime('%H:%M:%S')}][{step_name}]{RESET}"
    logging.info(header)
    _append_to_log(header)
    if messages:
        for message in messages:
            line = f"{colorCode}{message}{RESET}"
            print(line)
            _append_to_log(line)
