import importlib
import json
import sys
import time
import types
from statistics import mean
from unittest.mock import patch

from PIL import Image


def install_dependency_stubs():
    if "ollama" not in sys.modules:
        ollama_stub = types.ModuleType("ollama")
        ollama_stub.list = lambda: {"models": [{"name": "llama3"}]}
        ollama_stub.chat = lambda **kwargs: {"message": {"content": "[]"}}
        sys.modules["ollama"] = ollama_stub

    if "pytesseract" not in sys.modules:
        pyt_stub = types.ModuleType("pytesseract")
        pyt_stub.pytesseract = types.SimpleNamespace(tesseract_cmd="")
        pyt_stub.image_to_string = lambda image: "Paracetamol 500mg Warfarin 5mg"
        sys.modules["pytesseract"] = pyt_stub


def run_benchmark():
    install_dependency_stubs()

    sys.modules.pop("ocr_utils", None)
    sys.modules.pop("symptom", None)
    ocr_utils = importlib.import_module("ocr_utils")

    from med_db import MED_DB, check_interactions, extract_medicines_from_text
    from risk_engine import calculate_risk_score
    from symptom import analyze_symptom

    report = {
        "generated_at_epoch": time.time(),
        "benchmarks": {},
        "stability": {},
    }

    # OCR benchmark with varying image sizes
    ocr_results = []
    for side in [128, 512, 1024]:
        img = Image.new("RGB", (side, side), color="white")
        runs = []
        for _ in range(5):
            start = time.perf_counter()
            _ = ocr_utils.extract_text_from_image(img)
            runs.append((time.perf_counter() - start) * 1000)
        ocr_results.append({"image_px": f"{side}x{side}", "avg_ms": round(mean(runs), 3), "max_ms": round(max(runs), 3)})
    report["benchmarks"]["ocr_processing"] = ocr_results

    # Fuzzy matching benchmark with varying input sizes
    fuzzy_results = []
    for token_count in [100, 1000, 5000]:
        text = " ".join(["paracetamol", "warfarin", "aspirin", "nonsense"] * (token_count // 4))
        runs = []
        for _ in range(5):
            start = time.perf_counter()
            _ = extract_medicines_from_text(text)
            runs.append((time.perf_counter() - start) * 1000)
        fuzzy_results.append(
            {"tokens": token_count, "avg_ms": round(mean(runs), 3), "max_ms": round(max(runs), 3)}
        )
    report["benchmarks"]["fuzzy_matching"] = fuzzy_results

    # AI inference benchmark: same prompt (cache should help) vs unique prompts
    ai_same_prompt_runs = []
    with patch.object(ocr_utils.ollama, "chat", return_value={"message": {"content": "[]"}}):
        if hasattr(ocr_utils, "clear_medicine_extraction_cache"):
            ocr_utils.clear_medicine_extraction_cache()
        for _ in range(20):
            start = time.perf_counter()
            _ = ocr_utils.extract_medicines_with_llm("Paracetamol 500 mg once daily")
            ai_same_prompt_runs.append((time.perf_counter() - start) * 1000)

    ai_unique_prompt_runs = []
    with patch.object(ocr_utils.ollama, "chat", return_value={"message": {"content": "[]"}}):
        if hasattr(ocr_utils, "clear_medicine_extraction_cache"):
            ocr_utils.clear_medicine_extraction_cache()
        for i in range(20):
            start = time.perf_counter()
            _ = ocr_utils.extract_medicines_with_llm(f"Paracetamol 500 mg once daily {i}")
            ai_unique_prompt_runs.append((time.perf_counter() - start) * 1000)

    report["benchmarks"]["ai_inference"] = {
        "same_prompt_avg_ms": round(mean(ai_same_prompt_runs), 3),
        "unique_prompt_avg_ms": round(mean(ai_unique_prompt_runs), 3),
    }

    # Risk engine benchmark with varying medicine combination sizes
    med_keys = list(MED_DB.keys())
    risk_results = []
    for med_count in [2, 6, min(10, len(med_keys))]:
        meds = med_keys[:med_count]
        runs = []
        for _ in range(50):
            start = time.perf_counter()
            _ = calculate_risk_score(
                age=72,
                severity="High",
                medicines=meds,
                chronic_conditions=["Heart Disease", "Kidney Disease", "Hypertension"],
            )
            runs.append((time.perf_counter() - start) * 1000)
        risk_results.append({"med_count": med_count, "avg_ms": round(mean(runs), 3), "max_ms": round(max(runs), 3)})
    report["benchmarks"]["risk_score"] = risk_results

    # Stability scenarios
    sequential_start = time.perf_counter()
    for idx in range(200):
        complex_meds = med_keys[: min(len(med_keys), 8)]
        _ = check_interactions(complex_meds)
        _ = calculate_risk_score(
            age=30 + (idx % 40),
            severity=["Low", "Medium", "High", "Critical"][idx % 4],
            medicines=complex_meds,
            chronic_conditions=["Diabetes", "Hypertension"] if idx % 2 == 0 else [],
        )
        _ = analyze_symptom(" ".join(["severe headache with nausea"] * 50))
        _ = extract_medicines_from_text(" ".join(complex_meds * 100))
    sequential_elapsed = (time.perf_counter() - sequential_start) * 1000

    report["stability"] = {
        "sequential_requests_count": 200,
        "sequential_total_ms": round(sequential_elapsed, 3),
        "large_symptom_length": len(" ".join(["severe headache with nausea"] * 50)),
        "complex_medicine_count": min(len(med_keys), 8),
    }

    return report


if __name__ == "__main__":
    results = run_benchmark()
    print(json.dumps(results, indent=2))
