from datasets import load_dataset
import spacy
from collections import Counter
import re

nlp = spacy.load("en_core_web_sm")

def remove_urls_emails_symbols(text):
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def clean_with_spacy(texts, batch_size=50, n_process=1):
    cleaned = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and not token.like_num
        ]
        cleaned.append(" ".join(tokens))
    return cleaned

if __name__ == "__main__":
    dataset = load_dataset("ag_news")

    print("Number of rows (train):", len(dataset["train"]))
    print("Number of rows (test):", len(dataset["test"]))
    print("Columns:", dataset["train"].column_names)

    labels = ["World", "Sports", "Business", "Sci/Tech"]
    print("Categories:", labels)

    sample_texts = [d["text"] for d in dataset["train"].select(range(5))]
    basic_cleaned = [remove_urls_emails_symbols(t) for t in sample_texts]
    spacy_cleaned = clean_with_spacy(basic_cleaned)

    print("\n--- First sample texts ---")
    for raw, clean in zip(sample_texts, spacy_cleaned):
        print(f"Raw:   {raw[:80]}...")
        print(f"Clean: {clean[:80]}...\n")

    sample_texts = [d["text"] for d in dataset["train"].select(range(1000))]
    basic_cleaned = [remove_urls_emails_symbols(t) for t in sample_texts]
    cleaned_texts = clean_with_spacy(basic_cleaned)

    all_words = " ".join(cleaned_texts).split()
    word_freq = Counter(all_words)

    print("--- 10 most frequent words ---")
    for word, freq in word_freq.most_common(10):
        print(f"{word}: {freq}")
