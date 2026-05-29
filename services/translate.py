import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLLB-200 non-LLM translation
# ---------------------------------------------------------------------------

NLLB_MODEL_NAME = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
_TARGET_LANG = "eng_Latn"

# Lazy-loaded model cache
_nllb_tokenizer = None
_nllb_model = None

# Mapping from langdetect ISO 639-1 codes to NLLB BCP-47+ language codes
_LANGDETECT_TO_NLLB: dict[str, str] = {
    "af": "afr_Latn", "ar": "arb_Arab", "bg": "bul_Cyrl",
    "bn": "ben_Beng", "ca": "cat_Latn", "cs": "ces_Latn",
    "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn",
    "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn",
    "et": "est_Latn", "fa": "pes_Arab", "fi": "fin_Latn",
    "fr": "fra_Latn", "gu": "guj_Gujr", "he": "heb_Hebr",
    "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn",
    "hy": "hye_Armn", "id": "ind_Latn", "it": "ita_Latn",
    "ja": "jpn_Jpan", "ka": "kat_Geor", "ko": "kor_Hang",
    "lt": "lit_Latn", "lv": "lvs_Latn", "mk": "mkd_Cyrl",
    "ml": "mal_Mlym", "mr": "mar_Deva", "ms": "zsm_Latn",
    "mt": "mlt_Latn", "nl": "nld_Latn", "no": "nob_Latn",
    "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn",
    "ru": "rus_Cyrl", "sk": "slk_Latn", "sl": "slv_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu",
    "th": "tha_Thai", "tl": "tgl_Latn", "tr": "tur_Latn",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "vi": "vie_Latn",
    "zh-cn": "zho_Hans", "zh-tw": "zho_Hant",
}


def _load_nllb():
    """Lazily load the NLLB tokenizer and model (cached after first call)."""
    global _nllb_tokenizer, _nllb_model
    if _nllb_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info("Loading NLLB model '%s' …", NLLB_MODEL_NAME)
            _nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
            _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
            logger.info("NLLB model loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load NLLB model '{NLLB_MODEL_NAME}': {e}") from e
    return _nllb_tokenizer, _nllb_model


def translate_question(question: str, llm, use_llm: bool = False) -> str:
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
            "If the question is already in English, return it unchanged. "
            "If it is not in English, translate it into English. "
            "Do not add, remove, or rephrase any content. "
            "Only output the English sentence without any additional explanation.\n\n"
            f"Question:\n{text}"
        )

        user_prompt_new = (
            "Rewrite the following question to make the meaning clearer and more concise. "
            "Maintain the original meaning and intent of the question! "
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

def _translate_with_nllm(question: str) -> str:
    """
    Translate *question* into English using Facebook's NLLB-200 model.

    The source language is detected automatically with langdetect.
    If the question is already in English, it is returned unchanged.
    If language detection or translation fails, the original question is
    returned so the pipeline can continue gracefully.

    The NLLB model is lazy-loaded on the first call and cached in memory
    for subsequent calls. Use the ``NLLB_MODEL`` environment variable to
    override the default model (``facebook/nllb-200-distilled-600M``).
    """
    from langdetect import detect, LangDetectException

    if not question:
        return question

    # --- detect source language -------------------------------------------
    try:
        src_iso = detect(question)
    except LangDetectException:
        logger.warning("Language detection failed for question; returning as-is.")
        return question

    # already English — nothing to do
    if src_iso == "en":
        return question

    # --- map to NLLB language code ----------------------------------------
    src_nllb = _LANGDETECT_TO_NLLB.get(src_iso)
    if src_nllb is None:
        logger.warning(
            "Unknown language '%s' detected; no NLLB code mapping. Returning original.", src_iso
        )
        return question

    # --- translate -----------------------------------------------------------
    tokenizer, model = _load_nllb()
    tokenizer.src_lang = src_nllb
    inputs = tokenizer(question, return_tensors="pt")
    target_token_id = tokenizer.convert_tokens_to_ids(_TARGET_LANG)
    output_tokens = model.generate(**inputs, forced_bos_token_id=target_token_id)
    translated = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    logger.debug("NLLB translated [%s→en]: %r → %r", src_iso, question, translated)
    return translated
