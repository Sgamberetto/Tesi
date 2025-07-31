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
