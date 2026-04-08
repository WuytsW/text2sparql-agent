import logging


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

def log_message(step_name: str, color: str = "White", messages: list = None):
    colorCode = COLOR_MAP.get(color, WHITE)
    logging.info(f"{colorCode}[{step_name}]{RESET}")
    if messages:
        for message in messages:
            print(f"{colorCode}{message}{RESET}")