
import json
import torch
import os
import language_tool_python
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

# === CARICAMENTO DATI ===
with open("entries_diario_processate.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

# === MODELLI NLP ===
tokenizer = AutoTokenizer.from_pretrained("MilaNLProc/feel-it-italian")
model = AutoModelForSequenceClassification.from_pretrained("MilaNLProc/feel-it-italian")
tool = language_tool_python.LanguageTool('it')

# === FUNZIONI ===
def extract_text_sections(entry):
    fields = ["emozione", "pensieri", "descrizione_sensazioni", "note_aggiuntive"]
    texts = []
    for field in fields:
        section = entry.get(field, {})
        tokens = section.get("tokens", [])
        texts.append(" ".join(tokens))
    return " ".join(texts)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    labels = ["negative", "neutral", "positive", "mixed"]
    return dict(zip(labels, scores.tolist()))

def analyze_syntax(text):
    matches = tool.check(text)
    return len(matches), len(text.split())

def aggregate_emotions(entry):
    emos = entry.get("emozione", {}).get("lemmatized", [])
    return [e.lower() for e in emos]

# === RACCOLTA TESTI ===
texts = [extract_text_sections(entry) for entry in entries]
sentiments = [predict_sentiment(text) for text in texts]
syntax_errors = [analyze_syntax(text) for text in texts]
emotion_counter = Counter(e for entry in entries for e in aggregate_emotions(entry))

# === TOPIC MODELING ===
vectorizer = TfidfVectorizer(max_features=1000, stop_words="italian")
X = vectorizer.fit_transform(texts)

n_topics = 4
kmeans = KMeans(n_clusters=n_topics, random_state=42)
labels = kmeans.fit_predict(X)

# parole chiave per topic
keywords = {}
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(n_topics):
    keywords[i] = [terms[ind] for ind in order_centroids[i, :10]]

# media sentiment per topic
topic_sentiment = defaultdict(lambda: defaultdict(float))
topic_counts = Counter(labels)
for i, label in enumerate(labels):
    for k, v in sentiments[i].items():
        topic_sentiment[label][k] += v
for topic in topic_sentiment:
    for k in topic_sentiment[topic]:
        topic_sentiment[topic][k] /= topic_counts[topic]

# === VISUALIZZAZIONE ===

# Media generale sentimenti
avg_sent = defaultdict(float)
for s in sentiments:
    for k, v in s.items():
        avg_sent[k] += v
for k in avg_sent:
    avg_sent[k] /= len(sentiments)

# Sentiment globale
plt.figure(figsize=(6,4))
sns.barplot(x=list(avg_sent.keys()), y=list(avg_sent.values()), palette="coolwarm")
plt.title("Sentiment medio sui diari")
plt.tight_layout()
plt.savefig("sentiment_media.png")
plt.close()

# Emozioni
plt.figure(figsize=(6,4))
labels_, values_ = zip(*emotion_counter.most_common(10))
sns.barplot(x=values_, y=labels_, palette="viridis")
plt.title("Frequenza emozioni nominali")
plt.tight_layout()
plt.savefig("frequenza_emozioni.png")
plt.close()

# Sentiment per topic
plt.figure(figsize=(10,6))
topic_names = [f"Topic {i}" for i in range(n_topics)]
for sentiment_class in ["negative", "neutral", "positive", "mixed"]:
    values = [topic_sentiment[i][sentiment_class] for i in range(n_topics)]
    sns.lineplot(x=topic_names, y=values, label=sentiment_class)
plt.title("Sentiment medio per topic")
plt.ylabel("Probabilità")
plt.tight_layout()
plt.savefig("sentiment_per_topic.png")
plt.close()

# === REPORT ===
tot_errors = sum(e for e, _ in syntax_errors)
tot_words = sum(w for _, w in syntax_errors)
syntax_rate = (tot_errors / tot_words) * 100 if tot_words > 0 else 0

with open("report_risultati.txt", "w", encoding="utf-8") as f:
    f.write(f"Numero totale di errori sintattici: {tot_errors}\n")
    f.write(f"Numero totale di parole: {tot_words}\n")
    f.write(f"Tasso di errore sintattico: {syntax_rate:.2f}%\n\n")
    f.write("Sentiment medio:\n")
    for k, v in avg_sent.items():
        f.write(f"  - {k}: {v:.3f}\n")
    f.write("\nTop emozioni:\n")
    for emo, count in emotion_counter.most_common(10):
        f.write(f"  - {emo}: {count}\n")
    f.write("\nTopic trovati:\n")
    for i in range(n_topics):
        f.write(f"\nTopic {i}:\n")
        f.write("Parole chiave: " + ", ".join(keywords[i]) + "\n")
        for k, v in topic_sentiment[i].items():
            f.write(f"  Sentiment {k}: {v:.3f}\n")

print("✅ Analisi completata con topic modelling. Report e grafici generati.")
