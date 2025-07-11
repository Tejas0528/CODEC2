import csv
import numpy as np

def load_data(filename):
    texts = []
    labels = []
    with open(filename, "r", encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            texts.append(row['tweet'].lower())
            labels.append(1 if row['sentiment'].strip() == "positive" else 0)
    return texts, np.array(labels)

def build_vocab(texts):
    vocab = set()
    for text in texts:
        for word in text.split():
            vocab.add(word)
    return sorted(list(vocab))

def vectorize(texts, vocab):
    vectors = []
    for text in texts:
        vector = [text.split().count(word) for word in vocab]
        vectors.append(vector)
    return np.array(vectors)

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.vocab_size = X.shape[1]
        self.class_probs = {}
        self.word_probs = {}

        for c in self.classes:
            X_c = X[y == c]
            self.class_probs[c] = len(X_c) / len(X)
            self.word_probs[c] = (np.sum(X_c, axis=0) + 1) / (np.sum(X_c) + self.vocab_size)

    def predict(self, X):
        results = []
        for row in X:
            probs = {}
            for c in self.classes:
                log_prob = np.log(self.class_probs[c])
                log_prob += np.sum(np.log(self.word_probs[c]) * row)
                probs[c] = log_prob
            results.append(max(probs, key=probs.get))
        return np.array(results)

texts, labels = load_data("twitter_sentiment_data.csv")
vocab = build_vocab(texts)
X = vectorize(texts, vocab)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = labels[:split], labels[split:]

nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
