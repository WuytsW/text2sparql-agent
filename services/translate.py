import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv


def translate_question(question: str, llm) -> str:
    """
    Always sends question to the LLM.
    LLM decides whether to keep it or translate it to English.
    """

    if not question:
        raise ValueError("Empty question provided.")

    return _translate_with_llm(question, llm)


def _translate_with_llm(text: str, llm) -> str:
    """
    Sends the text to the LLM with smart prompt engineering.
    """


    try:
        user_prompt = (
            "You are a professional translator. "
            "If the question is already in English, simply return it unchanged. "
            "If it is not in English, translate it into English. "
            "Only output the English sentence without any additional explanation.\n\n"
            f"Question:\n{text}"
        )

        response = llm.invoke([
            SystemMessage(content="You are a translator for any given language into English."),
            HumanMessage(content=user_prompt)
        ])

        translated_text = response.content.strip()
        return translated_text

    except Exception as e:
        raise RuntimeError(f"Translation failed: {e}")
