import joblib
from preprocess_and_analysis import remove_urls_emails_symbols, clean_with_spacy

model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

labels = ["World", "Sports", "Business", "Sci/Tech"]

def predict_category(text: str):
    cleaned = remove_urls_emails_symbols(text)
    processed = clean_with_spacy([cleaned])  
    features = vectorizer.transform(processed)
    pred = model.predict(features)[0]
    return labels[pred]

if __name__ == "__main__":
    while True:
        user_input = input("Enter a news sentence (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        category = predict_category(user_input)
        print(f"Predicted category: {category}")