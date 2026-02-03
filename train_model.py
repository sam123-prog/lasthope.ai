import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

data = pd.read_csv("train.csv")  # or data.csv (check name)

# some datasets use 'text' and 'label'
# some use 'content' and 'generated'

if "text" in data.columns:
    X = data["text"]
else:
    X = data["content"]

if "label" in data.columns:
    y = data["label"]
else:
    y = data["generated"]

vectorizer = CountVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Real dataset model trained successfully!")