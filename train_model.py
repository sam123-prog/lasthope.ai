import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import re

# -----------------------------
# Load dataset
# -----------------------------

data = pd.read_csv("train.csv")   # or data.csv (rename if needed)

print("Dataset loaded!")
print("Columns:", data.columns)


# -----------------------------
# Handle different column names
# -----------------------------

if "text" in data.columns:
    X = data["text"]
else:
    X = data["content"]

if "label" in data.columns:
    y = data["label"]
else:
    y = data["generated"]


# -----------------------------
# Basic text cleaning
# -----------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)      # remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove symbols
    text = re.sub(r"\s+", " ", text)         # remove extra spaces
    return text.strip()


X = X.apply(clean_text)


# -----------------------------
# Remove empty rows
# -----------------------------

mask = X.str.len() > 0
X = X[mask]
y = y[mask]


print("Cleaned dataset size:", len(X))


# -----------------------------
# Vectorization (better settings)
# -----------------------------

vectorizer = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

X_vec = vectorizer.fit_transform(X)


# -----------------------------
# Train model
# -----------------------------

model = MultinomialNB()
model.fit(X_vec, y)


# -----------------------------
# Save model & vectorizer
# -----------------------------

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


print("✅ Model trained and saved successfully!")
