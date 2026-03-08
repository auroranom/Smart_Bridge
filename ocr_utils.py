from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from PIL import Image

try:
    import pytesseract  # type: ignore
    HAS_PYTESSERACT = True
except Exception:  # pragma: no cover - import guard
    pytesseract = None
    HAS_PYTESSERACT = False

try:
    import ollama  # type: ignore
    HAS_OLLAMA = True
except Exception:  # pragma: no cover - import guard
    ollama = None
    HAS_OLLAMA = False


# ─────────────────────────────────────────────
#  TESSERACT PATH CONFIG
# ─────────────────────────────────────────────
TESSERACT_PATHS = [
    os.getenv("MEDSAFE_TESSERACT_CMD", ""),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]


def configure_tesseract() -> str | None:
    """Returns configured tesseract path or None if unavailable."""
    if not HAS_PYTESSERACT:
        return None

    for path in TESSERACT_PATHS:
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
    return None


TESSERACT_CMD = configure_tesseract()


# ─────────────────────────────────────────────
#  AUTO-DETECT AVAILABLE OLLAMA MODEL
# ─────────────────────────────────────────────
def get_available_model() -> str:
    """Auto-detects the best available Ollama model or uses env override."""
    env_model = os.getenv("MEDSAFE_OLLAMA_MODEL", "").strip()
    if env_model:
        return env_model

    preferred = ["llama3", "llama3.2:1b", "llama3.2", "phi3:mini", "phi3", "mistral"]
    if not HAS_OLLAMA:
        return "llama3"

    try:
        models = ollama.list()
        available = [m["name"] for m in models.get("models", [])]
        for pref in preferred:
            for avail in available:
                if pref in avail:
                    return avail
        if available:
            return available[0]
    except Exception:
        pass
    return "llama3"


# Start with a safe default and resolve lazily to avoid startup stalls.
OLLAMA_MODEL = os.getenv("MEDSAFE_OLLAMA_MODEL", "").strip() or "llama3"
_MODEL_PROBED = False


def resolve_ollama_model() -> str:
    """
    Lazily resolves model selection.
    Set MEDSAFE_DISABLE_MODEL_PROBE=1 to skip startup probing.
    """
    global OLLAMA_MODEL, _MODEL_PROBED

    if os.getenv("MEDSAFE_OLLAMA_MODEL", "").strip():
        OLLAMA_MODEL = os.getenv("MEDSAFE_OLLAMA_MODEL", "").strip()
        _MODEL_PROBED = True
        return OLLAMA_MODEL

    if _MODEL_PROBED:
        return OLLAMA_MODEL

    if os.getenv("MEDSAFE_DISABLE_MODEL_PROBE", "1").strip() == "1":
        _MODEL_PROBED = True
        return OLLAMA_MODEL

    OLLAMA_MODEL = get_available_model()
    _MODEL_PROBED = True
    return OLLAMA_MODEL


def dependency_status() -> dict:
    """Dependency status for deployment diagnostics."""
    return {
        "has_pytesseract": HAS_PYTESSERACT,
        "tesseract_cmd": TESSERACT_CMD,
        "has_ollama": HAS_OLLAMA,
        "ollama_model": OLLAMA_MODEL,
        "model_probe_enabled": os.getenv("MEDSAFE_DISABLE_MODEL_PROBE", "1").strip() != "1",
        "ollama_host": os.getenv("OLLAMA_HOST", ""),
    }


# ─────────────────────────────────────────────
#  RAW OCR TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_text_from_image(image: Image.Image) -> str:
    """
    Extracts raw text from a given PIL Image using Tesseract OCR.
    Falls back to a readable error message if OCR backend is unavailable.
    """
    if not HAS_PYTESSERACT:
        return "Error extracting text: pytesseract dependency is not installed."

    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"


# ─────────────────────────────────────────────
#  LLM-DRIVEN MEDICINE EXTRACTION (JSON OUTPUT)
# ─────────────────────────────────────────────
def _fallback_extract_medicines(ocr_text: str) -> list:
    """Non-LLM fallback using local fuzzy scan."""
    from med_db import extract_medicines_from_text

    meds = extract_medicines_from_text(ocr_text or "")
    return [{"medicine": med.capitalize(), "salt": None} for med in meds]


def _extract_medicines_with_llm_uncached(ocr_text: str) -> list:
    """
    Uses Ollama model to parse OCR text and extract medicines.
    Falls back to fuzzy extraction if model backend is unavailable.
    """
    text = (ocr_text or "").strip()
    if not text:
        return []

    if not HAS_OLLAMA:
        return _fallback_extract_medicines(text)

    model_name = resolve_ollama_model()

    prompt = f"""You are a medical data extraction assistant.

From the following prescription text extracted via OCR, identify all medicine names and their active drug or salt components.

Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
Each item must have exactly two keys: "medicine" and "salt".
If the salt is unknown, use null.

OCR Text:
\"\"\"
{text}
\"\"\"

JSON Output:"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            fallback = _fallback_extract_medicines(text)
            return fallback

        medicines = json.loads(match.group())
        if isinstance(medicines, list):
            return medicines
        return _fallback_extract_medicines(text)

    except json.JSONDecodeError:
        return _fallback_extract_medicines(text)
    except Exception as e:
        fallback = _fallback_extract_medicines(text)
        if fallback:
            return fallback
        return [{"medicine": f"LLM Error: {e}", "salt": None}]


@lru_cache(maxsize=128)
def _extract_medicines_with_llm_cached(ocr_text: str) -> tuple:
    """Caches extraction to reduce duplicate model invocations."""
    extracted = _extract_medicines_with_llm_uncached(ocr_text)
    normalized = []
    for item in extracted:
        medicine = str(item.get("medicine", ""))
        salt = item.get("salt")
        normalized.append((medicine, salt))
    return tuple(normalized)


def extract_medicines_with_llm(ocr_text: str) -> list:
    cached = _extract_medicines_with_llm_cached((ocr_text or "").strip())
    return [{"medicine": medicine, "salt": salt} for medicine, salt in cached]


def clear_medicine_extraction_cache() -> None:
    _extract_medicines_with_llm_cached.cache_clear()


# ─────────────────────────────────────────────
#  VALIDATE EXTRACTED MEDICINES AGAINST DB
# ─────────────────────────────────────────────
def validate_medicines_against_db(extracted: list, med_db: dict) -> list:
    """
    Validates extracted medicine names against MED_DB using fuzzy matching.
    """
    from med_db import find_medicine

    validated = []
    for item in extracted:
        name = item.get("medicine", "")
        matched = find_medicine(name)
        validated.append(
            {
                "medicine": name,
                "salt": item.get("salt"),
                "matched_db_key": matched,
                "in_database": matched is not None,
            }
        )
    return validated


# ─────────────────────────────────────────────
#  FULL PIPELINE: IMAGE → JSON MEDICINES
# ─────────────────────────────────────────────
def full_prescription_pipeline(image: Image.Image) -> dict:
    raw_text = extract_text_from_image(image)
    extracted = extract_medicines_with_llm(raw_text)
    validated = validate_medicines_against_db(extracted, {})

    return {"raw_text": raw_text, "medicines": validated}
