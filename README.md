# 📰 AI Internship Challenge – Text Classification with AG News

This project is part of the **AI Internship Challenge**. It demonstrates a complete pipeline for text classification using the publicly available **AG News** dataset.

---

## 📚 Dataset: AG News

- **Name**: AG News Dataset
- **Source**: [Hugging Face AG News Dataset](https://huggingface.co/datasets/ag_news)
- **Categories**:
  - World
  - Sports
  - Business
  - Sci/Tech
- **Samples**: 120,000 training and 7,600 test samples

---

## 🛠️ Features Implemented

### 1. 📥 Data Preparation

- Loaded AG News dataset (CSV format)
- Cleaned text:
  - Removed special characters
  - Lowercased
  - Removed stopwords
  - Applied lemmatization
- Compared **spaCy vs NLTK** preprocessing time/performance

### 2. 📊 Exploratory Data Analysis (EDA)

- Displayed sample counts per category
- Visualized most frequent words
- Checked class distribution and balance

### 3. 🤖 Model Training

- Used **TF-IDF** vectorization
- Trained two models:
  - **Logistic Regression**
  - **Multinomial Naive Bayes**
- Split data into 80% train / 20% test
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

### 4. 🔮 Prediction Script

- Added a script/function to input custom text and predict its category

---

## ✨ Bonus Features

- ✅ Used **TF-IDF** instead of raw Bag-of-Words
- ✅ Compared **two models** (Logistic Regression & Naive Bayes)
- ✅ Plotted **Confusion Matrix**
- ✅ Created and used a local **virtual environment** (`.venv`)
- ✅ Timing comparison of **spaCy vs NLTK**

---

## 🧪 How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/test-classification.git
cd test-classification
Create and activate a virtual environment:

bash
Copy code
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the scripts in order:

bash
Copy code
python dataset_overview.py
python preprocess_and_analysis.py
python train_models.py
python predict.py
python spacy_vs_nltk_timing.py
```
