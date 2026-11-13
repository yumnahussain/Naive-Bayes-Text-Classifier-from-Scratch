"""
UIC CS 411 - Project 03: Naive Bayes Text Classifier from Scratch
"""

import os
import csv
import math
import re

CLASSES = ("positive", "negative")

V = set()
count_wc = {c: {} for c in CLASSES} 
N_c = {c: 0 for c in CLASSES} 
num_sentences = {c: 0 for c in CLASSES} 
P_c = {c: 0.0 for c in CLASSES} 


def tokenize(text):
    """Lowercase the text and split it into word tokens.

    Apostrophes are removed, so "don't", "don’t" and "dont"
    all become the same word "dont".
    A token is a run of letters/digits.
    Every occurrence is kept, so repeated words count more than once.
    """
    text = text.lower().replace("'", "").replace("’", "").replace("‘", "")
    return re.findall(r"[a-z0-9]+", text)


def load_dataset(path):
    """Read a text,label CSV and return a list of (text, label) pairs."""
    rows = []
    if not os.path.exists(path) and not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path) # project folder
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            text, label = ",".join(row[:-1]).strip(), row[-1].strip().lower()
            if label not in CLASSES: # skip first row
                continue
            rows.append((text, label))
    return rows


def train_model(train_file="data/train.csv"):
    """Train the Naive Bayes classifier on the training dataset."""
    V.clear()
    for c in CLASSES:
        count_wc[c] = {}
        N_c[c] = 0
        num_sentences[c] = 0

    data = load_dataset(train_file)
    for text, label in data:
        num_sentences[label] += 1
        for w in tokenize(text):
            V.add(w)
            count_wc[label][w] = count(w, label) + 1
            N_c[label] += 1

    total_sentences = len(data)
    for c in CLASSES:
        P_c[c] = num_sentences[c] / total_sentences if total_sentences else 0.0

    return {
        "V_size": len(V),
        "N_c": dict(N_c),
        "num_sentences": dict(num_sentences),
        "P_c": dict(P_c),
    }


def count(w, c):
    """count(w, c): how many times word w occurs in training sentences of class c."""
    return count_wc[c].get(w, 0)


def P_w_given_c(w, c):
    """P(w | c) = (count(w, c) + 1) / (N_c + |V|)   (Laplace smoothing, alpha = 1)."""
    return (count(w, c) + 1) / (N_c[c] + len(V))


def score(text):
    """score(c) = log P(c) + sum of log P(w | c) for each token w that is in V."""
    tokens = [w for w in tokenize(text) if w in V]   # ignore unknown words 
    scores = {}
    for c in CLASSES:
        s_c = math.log(P_c[c]) if P_c[c] > 0 else float("-inf")
        for w in tokens:
            s_c += math.log(P_w_given_c(w, c))
        scores[c] = s_c
    return scores


def predict(text):
    """
    Predict sentiment class for an input sentence.
    Returns: 'positive' or 'negative'
    Tie-breaking: If scores are tied, select the class with the higher prior P(c),
                  or default to 'positive' if the priors are equal.
    """
    if not V:
        train_model()
    scores = score(text)
    pos, neg = scores["positive"], scores["negative"]
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    if P_c["negative"] > P_c["positive"]:
        return "negative"
    return "positive"


def explain(text):
    """Print a step-by-step trace of a prediction (for the video walkthrough)."""
    tokens = tokenize(text)
    print(f"\nTokens: {tokens}")
    for c in CLASSES:
        print(f"  log P({c}) = log({P_c[c]:.4f}) = {math.log(P_c[c]):.4f}")
    for w in tokens:
        if w not in V:
            print(f"  '{w}': not in V -> ignored")
            continue
        parts = []
        for c in CLASSES:
            num = count(w, c) + 1
            den = N_c[c] + len(V)
            parts.append(f"P(w|{c[:3]}) = {num}/{den} = {num / den:.6f} "
                         f"(log {math.log(num / den):.4f})")
        print(f"  '{w}': " + " | ".join(parts))
    scores = score(text)
    print(f"  score(positive) = {scores['positive']:.4f}")
    print(f"  score(negative) = {scores['negative']:.4f}")
    print(f"  Prediction: {predict(text)}")


def evaluate_model(test_file="data/test.csv"):
    """
    Evaluate classifier performance on a test dataset.
    Returns:
        dict: A dictionary containing at least:
            {
                "accuracy": float,
                "confusion_matrix": {"TP": int, "TN": int, "FP": int, "FN": int}
            }
    """
    if not V:
        train_model()
    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    data = load_dataset(test_file)
    for text, label in data:
        pred = predict(text)
        if pred == "positive" and label == "positive":
            cm["TP"] += 1
        elif pred == "negative" and label == "negative":
            cm["TN"] += 1
        elif pred == "positive" and label == "negative":
            cm["FP"] += 1
        else:
            cm["FN"] += 1
    accuracy = (cm["TP"] + cm["TN"]) / len(data) if data else 0.0
    return {"accuracy": accuracy, "confusion_matrix": cm}


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    stats = train_model(os.path.join(base, "data", "train.csv"))

    print("=== Training summary ===")
    print(f"Sentences: {stats['num_sentences']}")
    print(f"Priors: P(positive) = {P_c['positive']:.4f}, "
          f"P(negative) = {P_c['negative']:.4f}")
    print(f"|V| = {stats['V_size']}")
    print(f"N_positive = {N_c['positive']}, N_negative = {N_c['negative']}")

    results = evaluate_model(os.path.join(base, "data", "test.csv"))
    cm = results["confusion_matrix"]
    print("\n=== Test set evaluation ===")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print("Confusion matrix (positive = positive class):")
    print(f"                 pred positive  pred negative")
    print(f"  actual positive   TP = {cm['TP']:<5}     FN = {cm['FN']}")
    print(f"  actual negative   FP = {cm['FP']:<5}     TN = {cm['TN']}")

    print("\nType a sentence to classify it (blank line or 'quit' to exit).")
    print("Start with 'count <word>' to see a word's counts in each class.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        if line.lower().startswith("count "):
            for w in tokenize(line[6:]):
                print(f"  '{w}': count(w, positive) = {count(w, 'positive')}, "
                      f"count(w, negative) = {count(w, 'negative')}, "
                      f"in V: {w in V}")
            continue
        explain(line)
