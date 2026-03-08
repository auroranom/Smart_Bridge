import importlib
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from PIL import Image


def install_dependency_stubs_if_missing():
    if "ollama" not in sys.modules:
        ollama_stub = types.ModuleType("ollama")
        ollama_stub.list = lambda: {"models": [{"name": "llama3"}]}
        ollama_stub.chat = lambda **kwargs: {
            "message": {
                "content": "Educational guidance only. This is not a diagnosis and not medical advice."
            }
        }
        sys.modules["ollama"] = ollama_stub

    if "pytesseract" not in sys.modules:
        pyt_stub = types.ModuleType("pytesseract")
        pyt_stub.pytesseract = types.SimpleNamespace(tesseract_cmd="")
        pyt_stub.image_to_string = lambda image: "Paracetamol 500mg\nWarfarin 5mg\nAspirin 75mg"
        sys.modules["pytesseract"] = pyt_stub


def run_e2e_validation() -> dict:
    install_dependency_stubs_if_missing()

    # Reload modules so fallback paths use current dependency availability
    for mod in ["ocr_utils", "symptom"]:
        sys.modules.pop(mod, None)
    ocr_utils = importlib.import_module("ocr_utils")
    symptom = importlib.import_module("symptom")

    from med_db import check_interactions
    from risk_engine import calculate_risk_score
    import session_logger

    report = {"checks": [], "pass": True}

    def record(name: str, condition: bool, details: str):
        report["checks"].append({"name": name, "passed": condition, "details": details})
        if not condition:
            report["pass"] = False

    # 1) Interaction workflow
    interaction_warnings = check_interactions(["warfarin", "aspirin", "ibuprofen"])
    record(
        "interaction_workflow",
        isinstance(interaction_warnings, list) and len(interaction_warnings) >= 1,
        f"warnings_found={len(interaction_warnings)}",
    )

    # 2) OCR + extraction workflow
    img = Image.new("RGB", (256, 256), color="white")
    raw_text = ocr_utils.extract_text_from_image(img)
    extracted = ocr_utils.extract_medicines_with_llm(raw_text)
    validated = ocr_utils.validate_medicines_against_db(extracted, {})
    record(
        "ocr_extraction_workflow",
        isinstance(raw_text, str) and isinstance(validated, list),
        f"raw_text_len={len(raw_text)}, validated_count={len(validated)}",
    )

    # 3) Symptom + AI explanation workflow
    symptom_guidance = symptom.analyze_symptom("fever")
    ai_explanation = symptom.ai_symptom_explanation("fever with chills")
    safe_text = "not a diagnosis" in ai_explanation.lower() or "not medical advice" in ai_explanation.lower()
    record(
        "symptom_ai_workflow",
        isinstance(symptom_guidance, str) and isinstance(ai_explanation, str) and safe_text,
        "AI response includes educational safety disclaimer",
    )

    # 4) Side-effect logging workflow
    with tempfile.TemporaryDirectory() as tmp:
        temp_log = str(Path(tmp) / "e2e_session_log.json")
        with patch.object(session_logger, "LOG_FILE", temp_log):
            session_logger.clear_session_log()
            side_effect_payload = {
                "medicines": ["Warfarin", "Aspirin"],
                "age": 68,
                "gender": "Male",
                "symptom": "bleeding gums and bruising",
            }
            session_logger.log_session_event("side_effect_check", side_effect_payload)
            summary = session_logger.get_session_summary()
            side_effect_events = summary.get("event_breakdown", {}).get("side_effect_check", 0)
    record(
        "side_effect_logging",
        side_effect_events >= 1,
        f"side_effect_check_events={side_effect_events}",
    )

    # 5) Risk workflow
    risk = calculate_risk_score(
        age=76,
        severity="Critical",
        medicines=["warfarin", "aspirin", "methotrexate"],
        chronic_conditions=["Heart Disease", "Kidney Disease"],
    )
    record(
        "risk_workflow",
        risk.get("category") in {"Low", "Medium", "High", "Critical"},
        f"category={risk.get('category')}, percentage={risk.get('percentage')}",
    )

    # 6) Fallback safety checks
    deps = ocr_utils.dependency_status()
    if not deps.get("has_pytesseract"):
        record(
            "ocr_fallback_safety",
            raw_text.lower().startswith("error extracting text:"),
            "OCR fallback emits readable error",
        )
    if not deps.get("has_ollama"):
        fallback_text = symptom.ai_doubt_solver("Is this dangerous?")
        record(
            "ai_fallback_safety",
            "not a diagnosis" in fallback_text.lower() or "not medical advice" in fallback_text.lower(),
            "AI fallback remains educational/non-diagnostic",
        )

    return report


if __name__ == "__main__":
    result = run_e2e_validation()
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)
