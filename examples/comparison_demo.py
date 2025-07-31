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
