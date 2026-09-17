import joblib

from preprocess import clean_text


# Load trained model and vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def predict_spam(message):
    # Clean message
    cleaned_message = clean_text(message)

    # Convert text to TF-IDF
    message_vector = vectorizer.transform([cleaned_message])

    # Predict
    prediction = model.predict(message_vector)[0]

    # Probability
    probability = model.predict_proba(message_vector)[0]

    return prediction, probability


# Test messages
messages = [
    "Hey, are you coming to college tomorrow?",
    "Congratulations! You have won a FREE cash prize! Click now!"
]

for message in messages:
    prediction, probability = predict_spam(message)

    print("\nMessage:", message)
    print("Prediction:", "SPAM" if prediction == 1 else "HAM")
    print("Probability:", probability)