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
