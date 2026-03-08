import unittest

from risk_engine import apply_safety_rules, calculate_risk_score


class TestRiskEngine(unittest.TestCase):
    def test_low_risk_case(self):
        result = calculate_risk_score(
            age=30,
            severity="Low",
            medicines=[],
            chronic_conditions=[],
        )
        self.assertEqual(result["category"], "Low")
        self.assertGreaterEqual(result["percentage"], 0)
        self.assertLessEqual(result["percentage"], 100)

    def test_critical_case(self):
        result = calculate_risk_score(
            age=80,
            severity="Critical",
            medicines=["warfarin", "aspirin", "methotrexate"],
            chronic_conditions=["Heart Disease", "Kidney Disease", "Cancer"],
        )
        self.assertEqual(result["category"], "Critical")
        self.assertGreater(result["percentage"], 70)

    def test_safety_rules_trigger(self):
        alerts = apply_safety_rules(age=10, medicines=["aspirin"], severity="Low")
        self.assertTrue(any("contraindicated" in item.lower() for item in alerts))


if __name__ == "__main__":
    unittest.main()
