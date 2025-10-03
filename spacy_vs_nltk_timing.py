from datasets import load_dataset
import spacy
import time

dataset = load_dataset("ag_news")
sample_texts = [d["text"] for d in dataset["train"].select(range(2000))]

nlp = spacy.load("en_core_web_sm")

def clean_texts_loop(texts):
    cleaned = []
    for text in texts:
        doc = nlp(text.lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and not token.like_num
        ]
        cleaned.append(" ".join(tokens))
    return cleaned

def clean_texts_pipe(texts, batch_size=50, n_process=1):
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
    start = time.time()
    _ = clean_texts_loop(sample_texts)
    end = time.time()
    print(f"Time (for loop): {end - start:.2f} seconds")

    start = time.time()
    _ = clean_texts_pipe(sample_texts)
    end = time.time()
    print(f"Time (nlp.pipe): {end - start:.2f} seconds")
