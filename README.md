# Buginator: GitHub Bug Severity Classification using NLP

## Overview

Buginator is a Natural Language Processing (NLP) application designed to automatically classify GitHub issue reports into two severity categories:

* **Critical**
* **Non-Critical**

The project aims to assist software development teams in prioritizing bug reports more efficiently by reducing the need for manual issue triaging. Using classical machine learning techniques and text processing pipelines, Buginator can predict bug severity directly from the textual content of GitHub issues.

## Features

* Automated bug severity prediction
* Real-time classification through a Streamlit web application
* Advanced text preprocessing pipeline for GitHub issue reports
* Comparison of multiple machine learning algorithms
* Performance visualization and model evaluation dashboard

---

## Dataset

The project uses the **GitHub Issues Dataset** from Hugging Face, containing more than **114,000 issue reports** collected from public GitHub repositories.

Original severity labels:

* Critical
* Major
* Minor

For this research, the labels were transformed into a binary classification problem:

| Original Label | New Label    |
| -------------- | ------------ |
| Critical       | Critical     |
| Major          | Non-Critical |
| Minor          | Non-Critical |

Final dataset size:

* **112,264 issue reports**
* Duplicate records removed
* Naturally imbalanced distribution

---

## Text Preprocessing Pipeline

The issue reports undergo extensive preprocessing before model training:

1. Lowercasing
2. URL removal
3. Email removal
4. Code snippet removal
5. File path removal
6. Stack trace removal
7. HTML tag removal
8. Mention and hashtag removal
9. Special character filtering
10. Whitespace normalization
11. Tokenization
12. Stopword removal
13. Lemmatization

The goal is to remove GitHub-specific noise while preserving severity-related information.

---

## Feature Engineering

Text data is converted into numerical representations using **TF-IDF Vectorization** with:

* Unigrams
* Bigrams
* Trigrams

Configuration:

```python
max_features = 10000
ngram_range = (1, 3)
min_df = 5
max_df = 0.8
```

---

## Machine Learning Models

Three classical machine learning algorithms were evaluated:

### Logistic Regression (LR)

* Class imbalance handling using `class_weight='balanced'`
* Strong regularization (`C=0.1`)

### Multinomial Naive Bayes (MNB)

* Fast baseline classifier
* Optimized smoothing parameter

### Support Vector Machine (SVM)

* Linear kernel
* Designed for sparse, high-dimensional text data
* Best overall performance

---

## Results

### Model Comparison

| Model                   | Accuracy   | Critical F1 |
| ----------------------- | ---------- | ----------- |
| Logistic Regression     | 85.66%     | 0.87        |
| Multinomial Naive Bayes | 78.16%     | 0.79        |
| Support Vector Machine  | **93.48%** | **0.94**    |

### Best Model: SVM

| Class        | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| Critical     | 0.99      | 0.90   | 0.94     |
| Non-Critical | 0.87      | 0.99   | 0.93     |

The Support Vector Machine achieved the strongest performance, demonstrating excellent capability in handling sparse TF-IDF representations and imbalanced bug severity data.

---

## System Architecture

```text
User Input
     │
     ▼
Text Preprocessing
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Trained SVM Model
     │
     ▼
Severity Prediction
(Critical / Non-Critical)
```

---

## Technology Stack

* Python
* Scikit-Learn
* NLTK
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Streamlit

---

## Future Improvements

Potential future developments include:

* Fine-tuning transformer models such as BERT or RoBERTa
* Multi-class severity classification
* Integration of GitHub metadata features
* GitHub Actions or CI/CD deployment for automatic issue triaging
* Repository-specific severity prediction models

---

## Team Members

| Name                      | Student ID |
| ------------------------- | ---------- |
| Angelina Jolie Candaya    | 2802541644 |
| Joshua Kevin Liem         | 2802535572 |
| Maureen Calista Surjo     | 2802536392 |
| Nicholas Hubert Soegihono | 2802515564 |
| Zahra Zakiyyah Priyono    | 2802492554 |

---

## Project Outcome

This project demonstrates that classical NLP techniques combined with proper preprocessing and feature engineering can achieve high-performance bug severity classification without requiring computationally expensive deep learning models.

The final SVM model achieved a **93.48% accuracy** and **0.94 F1-score** on Critical bug detection, making it suitable for supporting real-world software issue triaging workflows.
