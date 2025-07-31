# File: italian_nlp/__init__.py
"""
Italian Text Processing Library

A simple yet powerful library for Italian text processing using NLTK and spaCy.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .processor import ItalianTextProcessor
from .utils import get_frequency_analysis, compare_processing_methods

__all__ = [
    "ItalianTextProcessor",
    "get_frequency_analysis", 
    "compare_processing_methods"
]

# ============================================================================
# File: italian_nlp/processor.py
"""
Main processor class for Italian text processing.
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from typing import List, Dict, Optional, Tuple
import logging

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ItalianTextProcessor:
    """
    A comprehensive Italian text processor using NLTK and spaCy.
    
    Features:
    - Tokenization with NLTK
    - Stemming with NLTK Snowball stemmer
    - Lemmatization with spaCy (fallback available)
    - Italian stopwords removal
    - POS tagging with spaCy
    """
    
    def __init__(self, load_spacy: bool = True):
        """
        Initialize the Italian text processor.
        
        Args:
            load_spacy (bool): Whether to load spaCy model for lemmatization
        """
        self.spacy_model = None
        self._setup_nltk()
        
        if load_spacy and SPACY_AVAILABLE:
            self._setup_spacy()
        
        self._setup_fallback_lemmatizer()
    
    def _setup_nltk(self) -> None:
        """Setup NLTK components."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        # Initialize NLTK tools
        self.stemmer = SnowballStemmer('italian')
        self.stop_words = set(stopwords.words('italian'))
        logger.info("NLTK components initialized")
    
    def _setup_spacy(self) -> None:
        """Setup spaCy Italian model."""
        try:
            self.spacy_model = spacy.load("it_core_news_sm")
            logger.info("spaCy Italian model loaded successfully")
        except OSError:
            logger.warning(
                "spaCy Italian model not found. "
                "Install with: python -m spacy download it_core_news_sm"
            )
            self.spacy_model = None
    
    def _setup_fallback_lemmatizer(self) -> None:
        """Setup fallback lemmatization dictionary."""
        self.lemma_dict = {
            # Common verb conjugations
            'sono': 'essere', 'sei': 'essere', 'è': 'essere', 'siamo': 'essere',
            'siete': 'essere', 'era': 'essere', 'erano': 'essere',
            'ho': 'avere', 'hai': 'avere', 'ha': 'avere', 'abbiamo': 'avere',
            'avete': 'avere', 'hanno': 'avere', 'aveva': 'avere', 'avevano': 'avere',
            'faccio': 'fare', 'fai': 'fare', 'fa': 'fare', 'facciamo': 'fare',
            'fate': 'fare', 'fanno': 'fare', 'faceva': 'fare', 'facevano': 'fare',
            'vado': 'andare', 'vai': 'andare', 'va': 'andare', 'andiamo': 'andare',
            'andate': 'andare', 'vanno': 'andare', 'andava': 'andare', 'andavano': 'andare',
            'dico': 'dire', 'dici': 'dire', 'dice': 'dire', 'diciamo': 'dire',
            'dite': 'dire', 'dicono': 'dire', 'diceva': 'dire', 'dicevano': 'dire',
            'vedo': 'vedere', 'vedi': 'vedere', 'vede': 'vedere', 'vediamo': 'vedere',
            'vedete': 'vedere', 'vedono': 'vedere', 'vedeva': 'vedere', 'vedevano': 'vedere',
            
            # Plural to singular nouns
            'case': 'casa', 'libri': 'libro', 'bambini': 'bambino', 'bambine': 'bambina',
            'uomini': 'uomo', 'donne': 'donna', 'ragazzi': 'ragazzo', 'ragazze': 'ragazza',
            'gatti': 'gatto', 'cani': 'cane', 'macchine': 'macchina', 'strade': 'strada',
            
            # Adjective forms
            'bella': 'bello', 'belle': 'bello', 'belli': 'bello',
            'grande': 'grande', 'grandi': 'grande',
            'piccola': 'piccolo', 'piccole': 'piccolo', 'piccoli': 'piccolo',
            'buona': 'buono', 'buone': 'buono', 'buoni': 'buono',
        }
    
    @property
    def spacy_available(self) -> bool:
        """Check if spaCy model is available."""
        return self.spacy_model is not None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Italian text using NLTK.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        if not text or not text.strip():
            return []
        
        # Use NLTK's word tokenizer
        tokens = word_tokenize(text.lower(), language='italian')
        
        # Remove punctuation and empty tokens
        tokens = [
            token for token in tokens 
            if token not in string.punctuation and token.strip()
        ]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove Italian stopwords.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Tokens with stopwords removed
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming using NLTK's Italian Snowball stemmer.
        
        Args:
            tokens (List[str]): List of tokens to stem
            
        Returns:
            List[str]: List of stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using spaCy or fallback dictionary.
        
        Args:
            tokens (List[str]): List of tokens to lemmatize
            
        Returns:
            List[str]: List of lemmatized tokens
        """
        if not tokens:
            return []
        
        if self.spacy_model:
            # Use spaCy for lemmatization
            text = " ".join(tokens)
            doc = self.spacy_model(text)
            return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        else:
            # Fallback to dictionary + stemming
            lemmas = []
            for token in tokens:
                token_lower = token.lower()
                if token_lower in self.lemma_dict:
                    lemmas.append(self.lemma_dict[token_lower])
                else:
                    lemmas.append(self.stemmer.stem(token_lower))
            return lemmas
    
    def get_pos_tags(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Get part-of-speech tags using spaCy.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Tuple[str, str, str]]: List of (token, pos, tag) tuples
        """
        if not self.spacy_model:
            return []
        
        doc = self.spacy_model(text.lower())
        return [
            (token.text, token.pos_, token.tag_) 
            for token in doc 
            if not token.is_punct and not token.is_space
        ]
    
    def process_text(self, text: str, remove_stopwords: bool = True) -> Dict:
        """
        Complete text processing pipeline.
        
        Args:
            text (str): Input text to process
            remove_stopwords (bool): Whether to remove stopwords
            
        Returns:
            Dict: Processing results with all analysis
        """
        if not text or not text.strip():
            return {
                'original_text': text,
                'all_tokens': [],
                'filtered_tokens': [],
                'stems': [],
                'lemmas': [],
                'pos_tags': [],
                'total_tokens': 0,
                'filtered_token_count': 0,
                'stopwords_removed': 0,
                'spacy_available': self.spacy_available
            }
        
        # Tokenization
        tokens = self.tokenize(text)
        original_tokens = tokens.copy()
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Stemming
        stems = self.stem(tokens)
        
        # Lemmatization
        lemmas = self.lemmatize_tokens(tokens)
        
        # POS tagging
        pos_tags = self.get_pos_tags(" ".join(tokens)) if tokens else []
        
        return {
            'original_text': text,
            'all_tokens': original_tokens,
            'filtered_tokens': tokens,
            'stems': stems,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'total_tokens': len(original_tokens),
            'filtered_token_count': len(tokens),
            'stopwords_removed': len(original_tokens) - len(tokens),
            'spacy_available': self.spacy_available
        }
    
    def compare_methods(self, text: str) -> Dict:
        """
        Compare different processing methods.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Comparison results
        """
        tokens = self.remove_stopwords(self.tokenize(text))
        
        return {
            'original_text': text,
            'tokens': tokens,
            'nltk_stems': self.stem(tokens),
            'spacy_lemmas': self.lemmatize_tokens(tokens),
            'pos_tags': self.get_pos_tags(text) if self.spacy_available else []
        }

# ============================================================================
# File: italian_nlp/utils.py
"""
Utility functions for Italian text processing.
"""

import nltk
from typing import Dict, List
from .processor import ItalianTextProcessor


def get_frequency_analysis(processor: ItalianTextProcessor, text: str) -> Dict:
    """
    Perform frequency analysis on Italian text.
    
    Args:
        processor (ItalianTextProcessor): Processor instance
        text (str): Input text
        
    Returns:
        Dict: Frequency analysis results
    """
    result = processor.process_text(text)
    
    # Calculate frequencies
    token_freq = nltk.FreqDist(result['filtered_tokens'])
    stem_freq = nltk.FreqDist(result['stems'])
    lemma_freq = nltk.FreqDist(result['lemmas'])
    
    return {
        'token_frequencies': dict(token_freq),
        'stem_frequencies': dict(stem_freq),
        'lemma_frequencies': dict(lemma_freq),
        'most_common_tokens': token_freq.most_common(10),
        'most_common_lemmas': lemma_freq.most_common(10)
    }


def compare_processing_methods(texts: List[str]) -> Dict:
    """
    Compare processing methods across multiple texts.
    
    Args:
        texts (List[str]): List of texts to analyze
        
    Returns:
        Dict: Comparison results
    """
    processor = ItalianTextProcessor()
    results = []
    
    for text in texts:
        comparison = processor.compare_methods(text)
        results.append(comparison)
    
    return {
        'individual_results': results,
        'summary': {
            'total_texts': len(texts),
            'spacy_available': processor.spacy_available,
            'avg_tokens_per_text': sum(len(r['tokens']) for r in results) / len(results)
        }
    }


def validate_italian_text(text: str) -> Dict:
    """
    Basic validation for Italian text characteristics.
    
    Args:
        text (str): Text to validate
        
    Returns:
        Dict: Validation results
    """
    processor = ItalianTextProcessor()
    result = processor.process_text(text, remove_stopwords=False)
    
    # Italian language indicators
    italian_articles = {'il', 'la', 'lo', 'gli', 'le', 'un', 'una', 'uno'}
    italian_prepositions = {'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra'}
    
    tokens_set = set(result['all_tokens'])
    articles_found = len(tokens_set.intersection(italian_articles))
    prepositions_found = len(tokens_set.intersection(italian_prepositions))
    
    return {
        'likely_italian': articles_found > 0 or prepositions_found > 0,
        'italian_articles_found': articles_found,
        'italian_prepositions_found': prepositions_found,
        'total_tokens': result['total_tokens'],
        'confidence_score': min((articles_found + prepositions_found) / 10, 1.0)
    }

# ============================================================================
# File: examples/basic_usage.py
"""
Basic usage examples for Italian Text Processing Library.
"""

from italian_nlp import ItalianTextProcessor

def main():
    """Demonstrate basic usage of the Italian text processor."""
    
    # Initialize processor
    print("Initializing Italian Text Processor...")
    processor = ItalianTextProcessor()
    print(f"spaCy available: {processor.spacy_available}\n")
    
    # Example texts
    texts = [
        "I bambini giocano felicemente nel parco con i loro amici.",
        "Le belle ragazze camminano per le strade della città italiana.",
        "Ho mangiato una pizza deliziosa al ristorante ieri sera.",
        "Gli studenti universitari studiano attentamente per gli esami."
    ]
    
    print("=== BASIC TEXT PROCESSING EXAMPLES ===\n")
    
    for i, text in enumerate(texts, 1):
        print(f"Example {i}: {text}")
        print("-" * 60)
        
        # Process text
        result = processor.process_text(text)
        
        print(f"Tokens: {result['filtered_tokens']}")
        print(f"Stems: {result['stems']}")
        print(f"Lemmas: {result['lemmas']}")
        print(f"Stats: {result['filtered_token_count']} tokens, "
              f"{result['stopwords_removed']} stopwords removed")
        
        if result['pos_tags']:
            print(f"POS Tags: {result['pos_tags'][:3]}...")  # Show first 3
        
        print()

if __name__ == "__main__":
    main()

# ============================================================================
# File: examples/comparison_demo.py
"""
Demonstration of stemming vs lemmatization comparison.
"""

from italian_nlp import ItalianTextProcessor

def main():
    """Compare stemming and lemmatization results."""
    
    processor = ItalianTextProcessor()
    
    print("=== STEMMING vs LEMMATIZATION COMPARISON ===\n")
    
    test_cases = [
        "I bambini giocavano felicemente con i loro giocattoli preferiti.",
        "Le macchine rosse correvano velocemente sulla strada principale.",
        "Gli studenti studiavano attentamente i libri di storia italiana.",
        "I gatti neri dormivano sui tetti delle case antiche."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}: {text}")
        print("-" * 70)
        
        comparison = processor.compare_methods(text)
        
        print(f"Tokens:       {comparison['tokens']}")
        print(f"NLTK Stems:   {comparison['nltk_stems']}")
        print(f"spaCy Lemmas: {comparison['spacy_lemmas']}")
        
        if comparison['pos_tags']:
            pos_summary = [(token, pos) for token, pos, _ in comparison['pos_tags']]
            print(f"POS Tags:     {pos_summary}")
        
        print()
        
        # Show differences
        stems = comparison['nltk_stems']
        lemmas = comparison['spacy_lemmas']
        
        if len(stems) == len(lemmas):
            differences = [
                (token, stem, lemma) 
                for token, stem, lemma in zip(comparison['tokens'], stems, lemmas)
                if stem != lemma
            ]
            
            if differences:
                print("Key Differences (Token -> Stem vs Lemma):")
                for token, stem, lemma in differences:
                    print(f"  {token} -> {stem} vs {lemma}")
                print()

if __name__ == "__main__":
    main()

# ============================================================================
# File: examples/interactive_demo.py
"""
Interactive demonstration of Italian text processing.
"""

from italian_nlp import ItalianTextProcessor, get_frequency_analysis

def main():
    """Interactive demo for testing custom Italian text."""
    
    print("=== INTERACTIVE ITALIAN TEXT PROCESSOR ===")
    print("Enter Italian text to analyze (or 'quit' to exit)")
    print("Commands: 'freq' for frequency analysis, 'compare' for method comparison")
    print("-" * 60)
    
    processor = ItalianTextProcessor()
    print(f"spaCy model loaded: {processor.spacy_available}")
    print()
    
    while True:
        user_input = input("Italian text > ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Arrivederci! 👋")
            break
        
        if not user_input:
            continue
        
        # Special commands
        if user_input.startswith('freq '):
            text = user_input[5:]
            print("\n--- FREQUENCY ANALYSIS ---")
            freq_analysis = get_frequency_analysis(processor, text)
            print(f"Most common tokens: {freq_analysis['most_common_tokens'][:5]}")
            print(f"Most common lemmas: {freq_analysis['most_common_lemmas'][:5]}")
            print()
            continue
        
        if user_input.startswith('compare '):
            text = user_input[8:]
            print("\n--- METHOD COMPARISON ---")
            comparison = processor.compare_methods(text)
            print(f"Tokens: {comparison['tokens']}")
            print(f"Stems:  {comparison['nltk_stems']}")
            print(f"Lemmas: {comparison['spacy_lemmas']}")
            print()
            continue
        
        # Regular processing
        print("\n--- PROCESSING RESULTS ---")
        result = processor.process_text(user_input)
        
        print(f"Tokens: {result['filtered_tokens']}")
        print(f"Stems:  {result['stems']}")
        print(f"Lemmas: {result['lemmas']}")
        
        if result['pos_tags']:
            pos_simple = [(token, pos) for token, pos, _ in result['pos_tags']]
            print(f"POS:    {pos_simple}")
        
        print(f"Stats:  {result['filtered_token_count']} tokens, "
              f"{result['stopwords_removed']} stopwords removed")
        print()

if __name__ == "__main__":
    main()

# ============================================================================
# File: tests/test_processor.py
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

# ============================================================================
# File: tests/test_utils.py
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

# ============================================================================
# File: tests/__init__.py
"""
Test package for Italian Text Processing Library.
"""

# ============================================================================
# File: setup.py
"""
Setup script for Italian Text Processing Library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="italian-text-processing",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple yet powerful Italian text processing library using NLTK and spaCy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/italian-text-processing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: Italian",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "italian-nlp-demo=examples.interactive_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# ============================================================================
# File: requirements.txt
nltk>=3.6
spacy>=3.4.0

# ============================================================================
# File: requirements-dev.txt
# Development dependencies
pytest>=6.0
pytest-cov>=2.0
flake8>=3.8.0
black>=21.0.0
isort>=5.0.0
mypy>=0.910

# Documentation
sphinx>=4.0.0
sphinx-rtd-theme>=0.5.0

# ============================================================================
# File: .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# spaCy models (downloaded separately)
*.model

# NLTK data (downloaded separately)
nltk_data/

# ============================================================================
# File: LICENSE
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# ============================================================================
# File: MANIFEST.in
include README.md
include LICENSE
include requirements.txt
include requirements-dev.txt
recursive-include examples *.py
recursive-include tests *.py
recursive-exclude * __pycache__
recursive-exclude * *.py[co]

# ============================================================================
# File: .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.7, 3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        python -m spacy download it_core_news_sm
    
    - name: Lint with flake8
      run: |
        flake8 italian_nlp --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 italian_nlp --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Format check with black
      run: |
        black --check italian_nlp/
    
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=italian_nlp --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

# ============================================================================
# File: pyproject.toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88
target-version = ['py37']
include = '\.pyi?
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["italian_nlp"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.7"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = ["nltk.*", "spacy.*"]
ignore_missing_imports = true