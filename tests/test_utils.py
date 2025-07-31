"""
Tests for utility functions.
"""

import unittest
from italian_nlp import ItalianTextProcessor
from italian_nlp.utils import get_frequency_analysis, validate_italian_text

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def setUp(self):
        """Set up test processor."""
        self.processor = ItalianTextProcessor()

    def test_frequency_analysis(self):
        """Test frequency analysis."""
        text = "Il gatto il gatto dorme dorme sempre."
        analysis = get_frequency_analysis(self.processor, text)

        # Check structure
        self.assertIn('token_frequencies', analysis)
        self.assertIn('lemma_frequencies', analysis)
        self.assertIn('most_common_tokens', analysis)

        # Check that repeated words have higher frequency
        token_freq = analysis['token_frequencies']
        self.assertGreater(token_freq.get('gatto', 0), 1)
        self.assertGreater(token_freq.get('dorme', 0), 1)

    def test_italian_text_validation(self):
        """Test Italian text validation."""
        italian_text = "Il gatto è sul tavolo con la famiglia."
        english_text = "The cat is on the table with the family."

        italian_result = validate_italian_text(italian_text)
        english_result = validate_italian_text(english_text)

        # Italian text should be detected as likely Italian
        self.assertTrue(italian_result['likely_italian'])
        self.assertGreater(italian_result['confidence_score'], 0)

        # English text should be less likely to be Italian
        self.assertLess(english_result['confidence_score'], italian_result['confidence_score'])


if __name__ == '__main__':
    unittest.main()
