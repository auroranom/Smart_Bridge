import ast
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitCallCollector(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
        ):
            self.calls.append((node.func.attr, node))
        self.generic_visit(node)


class TestUiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        collector = StreamlitCallCollector()
        collector.visit(cls.tree)
        cls.calls = collector.calls

    def _count(self, name: str) -> int:
        return sum(1 for call_name, _ in self.calls if call_name == name)

    def test_core_input_components_exist(self):
        self.assertGreaterEqual(self._count("text_input"), 1)
        self.assertGreaterEqual(self._count("text_area"), 1)
        self.assertGreaterEqual(self._count("file_uploader"), 1)
        self.assertGreaterEqual(self._count("selectbox"), 1)
        self.assertGreaterEqual(self._count("number_input"), 1)
        self.assertGreaterEqual(self._count("multiselect"), 1)
        self.assertGreaterEqual(self._count("button"), 1)

    def test_tabs_include_required_modules(self):
        tabs_calls = [node for name, node in self.calls if name == "tabs"]
        self.assertTrue(tabs_calls, "Expected st.tabs call in app.py")

        labels = []
        for node in tabs_calls:
            if node.args and isinstance(node.args[0], ast.List):
                for elt in node.args[0].elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        labels.append(elt.value)

        joined = " | ".join(labels)
        self.assertIn("Prescription OCR", joined)
        self.assertIn("Medicine Interaction Checker", joined)
        self.assertIn("Symptom & Doubt Solver", joined)
        self.assertIn("Side-Effect Monitor", joined)
        self.assertIn("Emergency Risk Predictor", joined)

    def test_feedback_and_visual_components_exist(self):
        self.assertGreaterEqual(self._count("success"), 1)
        self.assertGreaterEqual(self._count("warning"), 1)
        self.assertGreaterEqual(self._count("error"), 1)
        self.assertGreaterEqual(self._count("info"), 1)
        self.assertGreaterEqual(self._count("metric"), 1)
        self.assertGreaterEqual(self._count("expander"), 1)
        self.assertGreaterEqual(self._count("progress"), 1)


if __name__ == "__main__":
    unittest.main()
