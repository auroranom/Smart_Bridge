import unittest

from med_db import check_interactions


class TestCheckInteractions(unittest.TestCase):
    def test_detects_directional_interaction_independent_of_order(self):
        # "amoxicillin -> warfarin" exists in DB, but reverse does not.
        forward = check_interactions(["amoxicillin", "warfarin"])
        reverse = check_interactions(["warfarin", "amoxicillin"])

        self.assertTrue(forward, "Expected interaction for amoxicillin + warfarin")
        self.assertTrue(reverse, "Expected interaction for warfarin + amoxicillin")
        self.assertEqual(len(forward), 1)
        self.assertEqual(len(reverse), 1)

    def test_handles_fuzzy_names(self):
        # "warferin" should fuzzy-match to "warfarin"
        interactions = check_interactions(["warferin", "aspirin"])
        self.assertTrue(interactions)
        self.assertEqual(len(interactions), 1)

    def test_returns_empty_when_no_known_interaction(self):
        interactions = check_interactions(["cetirizine", "paracetamol"])
        self.assertEqual(interactions, [])

    def test_deduplicates_duplicate_inputs(self):
        interactions = check_interactions(["warfarin", "aspirin", "warfarin"])
        self.assertEqual(len(interactions), 1)


if __name__ == "__main__":
    unittest.main()
