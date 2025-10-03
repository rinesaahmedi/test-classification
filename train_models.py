from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from preprocess_and_analysis import remove_urls_emails_symbols, clean_with_spacy

if __name__ == "__main__":
    labels = ["World", "Sports", "Business", "Sci/Tech"]

    print("Loading dataset...")
    dataset = load_dataset("ag_news")

    train_subset = dataset["train"].select(range(10000))
    test_subset = dataset["test"].select(range(2000))

    print("Preprocessing text...")
    train_texts = [remove_urls_emails_symbols(d["text"]) for d in train_subset]
    test_texts = [remove_urls_emails_symbols(d["text"]) for d in test_subset]

    train_cleaned = clean_with_spacy(train_texts)
    test_cleaned = clean_with_spacy(test_texts)

    y_train = [d["label"] for d in train_subset]
    y_test = [d["label"] for d in test_subset]

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(train_cleaned)
    X_test = vectorizer.transform(test_cleaned)

    print("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=300, n_jobs=-1)
    log_reg.fit(X_train, y_train)
    y_pred_logreg = log_reg.predict(X_test)

    print("\n=== Logistic Regression ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
    print(classification_report(y_test, y_pred_logreg, target_names=labels))

    print("Training Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)

    print("\n=== Naive Bayes ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_nb))
    print(classification_report(y_test, y_pred_nb, target_names=labels))

    print("Plotting confusion matrix for Logistic Regression...")
    cm = confusion_matrix(y_test, y_pred_logreg)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("Saving models and vectorizer...")
    joblib.dump(log_reg, "logistic_regression_model.pkl")
    joblib.dump(nb_model, "naive_bayes_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

