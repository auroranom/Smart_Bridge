import importlib
import sys
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
        pyt_stub.image_to_string = lambda image: "Paracetamol 500mg"
        sys.modules["pytesseract"] = pyt_stub


class TestOcrUtilsModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_dependency_stubs()
        sys.modules.pop("ocr_utils", None)
        cls.ocr_utils = importlib.import_module("ocr_utils")

    def test_extract_text_from_image_success(self):
        img = Image.new("RGB", (20, 20), color="white")
        with patch.object(self.ocr_utils.pytesseract, "image_to_string", return_value="hello"):
            text = self.ocr_utils.extract_text_from_image(img)
        self.assertEqual(text, "hello")

    def test_extract_text_from_image_error(self):
        img = Image.new("RGB", (20, 20), color="white")
        with patch.object(self.ocr_utils.pytesseract, "image_to_string", side_effect=RuntimeError("boom")):
            text = self.ocr_utils.extract_text_from_image(img)
        self.assertIn("Error extracting text:", text)

    def test_extract_medicines_with_llm_parses_json(self):
        response = {
            "message": {
                "content": """```json
                [{"medicine":"Paracetamol","salt":"Acetaminophen"}]
                ```"""
            }
        }
        with patch.object(self.ocr_utils.ollama, "chat", return_value=response):
            medicines = self.ocr_utils.extract_medicines_with_llm("Paracetamol 500mg")

        self.assertIsInstance(medicines, list)
        self.assertEqual(medicines[0]["medicine"], "Paracetamol")

    def test_validate_medicines_against_db(self):
        extracted = [{"medicine": "Warfarin", "salt": "Warfarin"}]
        validated = self.ocr_utils.validate_medicines_against_db(extracted, {})
        self.assertEqual(len(validated), 1)
        self.assertIn("matched_db_key", validated[0])
        self.assertTrue(validated[0]["in_database"])


if __name__ == "__main__":
    unittest.main()
