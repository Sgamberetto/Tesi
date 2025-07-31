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
