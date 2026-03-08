import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import ocr_utils
import session_logger
import symptom


ROOT = Path(__file__).resolve().parents[1]


class TestReleaseReadiness(unittest.TestCase):
    def test_project_structure_core_files_exist(self):
        required = [
            ROOT / "app.py",
            ROOT / "med_db.py",
            ROOT / "ocr_utils.py",
            ROOT / "risk_engine.py",
            ROOT / "symptom.py",
            ROOT / "session_logger.py",
            ROOT / "requirements.txt",
            ROOT / "DEPLOYMENT.md",
            ROOT / "e2e_validate.py",
            ROOT / ".streamlit" / "config.toml",
            ROOT / ".streamlit" / "secrets.toml.example",
        ]
        missing = [str(path) for path in required if not path.exists()]
        self.assertEqual(missing, [], f"Missing release files: {missing}")

    def test_side_effect_logging_persists_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "session_log.json")
            with patch.object(session_logger, "LOG_FILE", log_path):
                session_logger.clear_session_log()
                payload = {
                    "medicines": ["Warfarin", "Aspirin"],
                    "age": 67,
                    "gender": "Male",
                    "symptom": "bruising",
                }
                session_logger.log_session_event("side_effect_check", payload)
                summary = session_logger.get_session_summary()
                self.assertEqual(summary["event_breakdown"].get("side_effect_check"), 1)

    def test_ocr_and_llm_fallback_paths(self):
        img = Image.new("RGB", (64, 64), color="white")

        with patch.object(ocr_utils, "HAS_PYTESSERACT", False):
            text = ocr_utils.extract_text_from_image(img)
            self.assertIn("Error extracting text:", text)

        with patch.object(ocr_utils, "HAS_OLLAMA", False):
            extracted = ocr_utils.extract_medicines_with_llm("Paracetamol Warfarin")
            self.assertIsInstance(extracted, list)
            self.assertGreaterEqual(len(extracted), 1)

    def test_symptom_ai_safe_fallback(self):
        with patch.object(symptom, "HAS_OLLAMA", False):
            response = symptom.ai_symptom_explanation("fever and cough")
            self.assertIn("Educational information only", response)
            self.assertIn("not a diagnosis", response.lower())

    def test_dependency_status_shape(self):
        deps = ocr_utils.dependency_status()
        for key in ["has_pytesseract", "tesseract_cmd", "has_ollama", "ollama_model", "ollama_host"]:
            self.assertIn(key, deps)


if __name__ == "__main__":
    unittest.main()
