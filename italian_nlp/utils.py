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
