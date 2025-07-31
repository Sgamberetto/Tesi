"""
Tests for the Italian text processor.
"""

import unittest
from italian_nlp import ItalianTextProcessor

class TestItalianTextProcessor(unittest.TestCase):
    """Test cases for ItalianTextProcessor."""

    def setUp(self):
        """Set up test processor."""
        self.processor = ItalianTextProcessor()

    def test_tokenization(self):
        """Test tokenization functionality."""
        text = "Ciao, come stai?"
        tokens = self.processor.tokenize(text)
        expected = ["ciao", "come", "stai"]
        self.assertEqual(tokens, expected)

    def test_stemming(self):
        """Test stemming functionality."""
        tokens = ["gatti", "bambini", "giocavano"]
        stems = self.processor.stem(tokens)
        # Check if stems are shorter than original words
        for token, stem in zip(tokens, stems):
            self.assertLessEqual(len(stem), len(token))

    def test_stopword_removal(self):
        """Test stopword removal."""
        tokens = ["il", "gatto", "è", "sul", "tavolo"]
        filtered = self.processor.remove_stopwords(tokens)
        self.assertIn("gatto", filtered)
        self.assertIn("tavolo", filtered)
        self.assertNotIn("il", filtered)
        self.assertNotIn("è", filtered)

    def test_lemmatization(self):
        """Test lemmatization functionality."""
        tokens = ["bambini", "giocavano"]
        lemmas = self.processor.lemmatize_tokens(tokens)
        self.assertEqual(len(lemmas), len(tokens))
        # Should return meaningful lemmas
        self.assertTrue(all(lemma.strip() for lemma in lemmas))

    def test_complete_processing(self):
        """Test complete processing pipeline."""
        text = "I bambini giocano nel parco."
        result = self.processor.process_text(text)

        # Check all expected keys are present
        expected_keys = [
            'original_text', 'all_tokens', 'filtered_tokens',
            'stems', 'lemmas', 'total_tokens', 'filtered_token_count',
            'stopwords_removed', 'spacy_available'
        ]

        for key in expected_keys:
            self.assertIn(key, result)

        # Check data types
        self.assertIsInstance(result['all_tokens'], list)
        self.assertIsInstance(result['filtered_tokens'], list)
        self.assertIsInstance(result['stems'], list)
        self.assertIsInstance(result['lemmas'], list)
        self.assertIsInstance(result['total_tokens'], int)
        self.assertIsInstance(result['filtered_token_count'], int)

    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.processor.process_text("")
        self.assertEqual(result['total_tokens'], 0)
        self.assertEqual(len(result['all_tokens']), 0)

    def test_pos_tagging(self):
        """Test POS tagging if spaCy is available."""
        if self.processor.spacy_available:
            text = "Il gatto dorme."
            pos_tags = self.processor.get_pos_tags(text)
            self.assertGreater(len(pos_tags), 0)
            # Each tag should be a tuple of (token, pos, tag)
            for tag_info in pos_tags:
                self.assertEqual(len(tag_info), 3)


if __name__ == '__main__':
    unittest.main()
