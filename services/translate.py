import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def translate_question(question: str, model_name: str = "openai/gpt-4o-mini") -> str:
    """
    Always sends question to the LLM.
    LLM decides whether to keep it or translate it to English.
    """
    if not question:
        raise ValueError("Empty question provided.")

    return _translate_with_llm(question, model_name)

def _translate_with_llm(text: str, model_name: str) -> str:
    """
    Sends the text to the LLM with smart prompt engineering.

    Args:
        text (str): The text to process.
        model_name (str): The model to use via OpenRouter.

    Returns:
        str: English version of the text.
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("mKGQAgent_Translation_LLM"),
        base_url="https://openrouter.ai/api/v1",
    )

    try:
        messages = [
            ("system", "You are a translator for any given language into English."),
            ("user", (
                "You are a professional translator. "
                "If the question is already in English, simply return it unchanged. "
                "If it is not in English, translate it into English. "
                "Only output the English sentence without any additional explanation.\n\n"
                f"Question:\n{text}"
            ))
        ]

        response = llm.invoke(messages)
        translated_text = response.content.strip()
        print("[INFO] Translation (or no-change) successful.")
        return translated_text

    except Exception as e:
        print(f"[ERROR] LLM translation failed: {e}")
        raise RuntimeError(f"Translation failed: {e}")
