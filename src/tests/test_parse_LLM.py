"""
    Test quotation extraction
"""
from unittest import TestCase
from paraphrase.utility.parse_LLM import _extract_quoted_phrase_indices


class Test(TestCase):
    def test_extract_quoted_phrase_indices(self):
        indices = _extract_quoted_phrase_indices(
            "Hello, this is a test. Hello, David.",
            "\"hello David\""
        )
        self.assertEqual([0, 0, 0, 0, 0, 1, 1], indices)

        indices = _extract_quoted_phrase_indices(
            "So that would be, actually, coming to New Jersey and being under the auspices, frankly, of De Lacy Davis.",
            "\"coming to New Jersey and being under the auspices\" \"of De Lacy Davis.\""
        )
        self.assertEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], indices)
        print(indices)

        indices = _extract_quoted_phrase_indices(
            "Hello, this is a test. Hello, David.",
            "\"Hello, David.\""
        )
        self.assertEqual([0, 0, 0, 0, 0, 1, 1], indices)

        indices = _extract_quoted_phrase_indices(
            "Hello, this is a test. Hello, David.",
            "\"Hello, David\""
        )
        self.assertEqual([0, 0, 0, 0, 0, 1, 1], indices)