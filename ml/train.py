import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text


# Load dataset
df = pd.read_csv("data/spam.csv")

# Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"])

# Clean text
df["text"] = df["text"].apply(clean_text)

# Features and labels
X = df["text"]
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
predictions = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nNaive Bayes model saved successfully!")