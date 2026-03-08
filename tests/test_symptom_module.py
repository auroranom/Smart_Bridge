import importlib
import sys
import types
import unittest
from unittest.mock import patch


def install_dependency_stubs():
    if "ollama" not in sys.modules:
        ollama_stub = types.ModuleType("ollama")
        ollama_stub.list = lambda: {"models": [{"name": "llama3"}]}
        ollama_stub.chat = lambda **kwargs: {"message": {"content": "Educational response."}}
        sys.modules["ollama"] = ollama_stub

    if "pytesseract" not in sys.modules:
        pyt_stub = types.ModuleType("pytesseract")
        pyt_stub.pytesseract = types.SimpleNamespace(tesseract_cmd="")
        pyt_stub.image_to_string = lambda image: "Paracetamol 500mg"
        sys.modules["pytesseract"] = pyt_stub


class TestSymptomModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_dependency_stubs()
        sys.modules.pop("ocr_utils", None)
        sys.modules.pop("symptom", None)
        cls.symptom = importlib.import_module("symptom")

    def test_analyze_symptom_known(self):
        result = self.symptom.analyze_symptom("headache")
        self.assertIn("Headache", result)

    def test_analyze_symptom_unknown(self):
        result = self.symptom.analyze_symptom("some_unknown_symptom_xyz")
        self.assertIn("not recognized", result.lower())

    def test_side_effect_analysis_returns_warnings(self):
        result = self.symptom.analyze_side_effects(
            medicines=["warfarin"],
            age=70,
            gender="Male",
            reported_symptom="bleeding and bruising",
        )
        self.assertTrue(any("high risk" in item.lower() or "consult your doctor" in item.lower() for item in result))

    def test_ai_symptom_prompt_and_disclaimer_guardrail(self):
        captured = {}

        def fake_chat(model, messages):
            captured["prompt"] = messages[0]["content"]
            return {"message": {"content": "Possible benign causes include dehydration."}}

        with patch.object(self.symptom.ollama, "chat", side_effect=fake_chat):
            output = self.symptom.ai_symptom_explanation("fever")

        self.assertIn("Do NOT diagnose", captured["prompt"])
        self.assertIn("not a diagnosis", output.lower())

    def test_ai_doubt_prompt_and_disclaimer_guardrail(self):
        captured = {}

        def fake_chat(model, messages):
            captured["prompt"] = messages[0]["content"]
            return {"message": {"content": "Mild post-vaccine fever can happen."}}

        with patch.object(self.symptom.ollama, "chat", side_effect=fake_chat):
            output = self.symptom.ai_doubt_solver("Is mild fever after vaccine normal?")

        self.assertIn("Educational only, not a diagnosis", captured["prompt"])
        self.assertIn("not a diagnosis", output.lower())


if __name__ == "__main__":
    unittest.main()
