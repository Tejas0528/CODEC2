# 🧠 Sentiment Analysis on Twitter Data (Pure Python)

## 📌 Project Overview

This project focuses on building a **Sentiment Analysis Model** using **pure Python and NumPy**, without relying on external libraries like `nltk` or `sklearn`. The goal is to classify tweets as **positive** or **negative** based on their content, using a **custom-built Naive Bayes Classifier** and a manual **Bag of Words** approach.

## ✅ Features

- Manual text preprocessing (lowercasing, tokenizing, removing punctuation)
- Custom vocabulary and Bag of Words implementation
- Naive Bayes classification without external ML libraries
- CSV file-based input/output for ease of data handling
- Lightweight and beginner-friendly structure

## 📂 Dataset

The dataset is a CSV file with the following format:

```csv
tweet,sentiment
I love this product,positive
This is awful,negative
...
