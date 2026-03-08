import importlib
import sys
import time
import types
import unittest
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


class TestPerformanceAndStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_dependency_stubs()
        sys.modules.pop("ocr_utils", None)
        sys.modules.pop("symptom", None)
        cls.ocr_utils = importlib.import_module("ocr_utils")
        cls.symptom = importlib.import_module("symptom")

        from med_db import MED_DB, check_interactions, extract_medicines_from_text
        from risk_engine import calculate_risk_score

        cls.med_db_keys = list(MED_DB.keys())
        cls.check_interactions_fn = staticmethod(check_interactions)
        cls.extract_medicines_from_text_fn = staticmethod(extract_medicines_from_text)
        cls.calculate_risk_score_fn = staticmethod(calculate_risk_score)

    def test_ai_medicine_extraction_cache_reduces_invocations(self):
        if hasattr(self.ocr_utils, "clear_medicine_extraction_cache"):
            self.ocr_utils.clear_medicine_extraction_cache()

        counter = {"calls": 0}

        def fake_chat(*args, **kwargs):
            counter["calls"] += 1
            return {"message": {"content": '[{"medicine":"Paracetamol","salt":"Acetaminophen"}]'}}

        with patch.object(self.ocr_utils.ollama, "chat", side_effect=fake_chat):
            for _ in range(20):
                result = self.ocr_utils.extract_medicines_with_llm("Paracetamol 500mg")
                self.assertTrue(result)

        self.assertEqual(counter["calls"], 1)

    def test_runtime_under_varying_input_sizes(self):
        sizes = [100, 1000, 5000]
        for token_count in sizes:
            with self.subTest(token_count=token_count):
                text = " ".join(["paracetamol", "warfarin", "aspirin", "nonsense"] * (token_count // 4))
                start = time.perf_counter()
                meds = self.extract_medicines_from_text_fn(text)
                elapsed = time.perf_counter() - start
                self.assertIn("paracetamol", meds)
                self.assertLess(elapsed, 1.5)

    def test_ocr_and_risk_smoke_performance(self):
        img = Image.new("RGB", (512, 512), color="white")
        start_ocr = time.perf_counter()
        text = self.ocr_utils.extract_text_from_image(img)
        ocr_elapsed = time.perf_counter() - start_ocr

        start_risk = time.perf_counter()
        for _ in range(300):
            _ = self.calculate_risk_score_fn(
                age=72,
                severity="High",
                medicines=self.med_db_keys[:8],
                chronic_conditions=["Heart Disease", "Kidney Disease", "Hypertension"],
            )
        risk_elapsed = time.perf_counter() - start_risk

        self.assertTrue(text)
        self.assertLess(ocr_elapsed, 1.0)
        self.assertLess(risk_elapsed, 2.0)

    def test_stability_sequential_requests(self):
        for idx in range(200):
            meds = self.med_db_keys[: min(8, len(self.med_db_keys))]
            interactions = self.check_interactions_fn(meds)
            risk = self.calculate_risk_score_fn(
                age=30 + (idx % 50),
                severity=["Low", "Medium", "High", "Critical"][idx % 4],
                medicines=meds,
                chronic_conditions=["Diabetes", "Hypertension"] if idx % 2 == 0 else [],
            )
            symptom_output = self.symptom.analyze_symptom(" ".join(["headache nausea"] * 40))
            fuzzy_output = self.extract_medicines_from_text_fn(" ".join(meds * 120))

            self.assertIsInstance(interactions, list)
            self.assertIn("category", risk)
            self.assertIsInstance(symptom_output, str)
            self.assertIsInstance(fuzzy_output, list)

    def test_large_symptom_description_and_complex_medicine_combo(self):
        large_symptom = " ".join(["persistent headache with nausea and dizziness"] * 300)
        output = self.symptom.analyze_symptom(large_symptom)
        self.assertIsInstance(output, str)

        complex_combo = self.med_db_keys
        interactions = self.check_interactions_fn(complex_combo)
        self.assertIsInstance(interactions, list)
        self.assertEqual(len(interactions), len(set(interactions)))


if __name__ == "__main__":
    unittest.main()
