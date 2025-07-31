import json
import nltk
from italian_nlp.processor import ItalianTextProcessor

processor = ItalianTextProcessor()

with open(r"C:\Users\Carlo\Downloads\Tesi\entries_diario.json", "r", encoding="utf-8") as f:
    data = json.load(f)

entries = data["diari_cognitivo_comportamentali"]

campi_testuali = [
    "situazione_spiacevole",
    "contesto",
    "aspetto_spiacevole",
    "comportamenti",
    "emozione",
    "descrizione_sensazioni",
    "pensieri",
    "note_aggiuntive"
]

risultati = []

for i, entry in enumerate(entries):
    print(f"\n Entry #{i+1} — ID: {entry.get('id')}")
    entry_risultato = {"id": entry.get("id")}

    for campo in campi_testuali:
        testo = entry.get(campo, "")
        if not testo or not isinstance(testo, str):
            continue

        print(f"\n Campo: {campo}")
        print(f" Testo: {testo}")

        tokens = [t for t in nltk.word_tokenize(testo, language="italian") if t.isalpha()]
        filtered = [t for t in tokens if t.lower() not in processor.stop_words]
        lemmatized = [processor.lemma_dict.get(t.lower(), t.lower()) for t in filtered]
        stemmed = [processor.stemmer.stem(t.lower()) for t in filtered]

        print("  ➤ Token:", tokens)
        print("  ➤ Senza stopwords:", filtered)
        print("  ➤ Lemmi (fallback):", lemmatized)
        print("  ➤ Stemming:", stemmed)

        entry_risultato[campo] = {
            "tokens": tokens,
            "filtered": filtered,
            "lemmatized": lemmatized,
            "stemmed": stemmed,
        }

    risultati.append(entry_risultato)

output_path = r"C:\Users\Carlo\Downloads\Tesi\analisi_diarioprocessati.json"
with open(output_path, "w", encoding="utf-8") as out_file:
    json.dump(risultati, out_file, ensure_ascii=False, indent=2)

print(f"\n Risultati salvati in: {output_path}")
