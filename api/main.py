from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

from ml.preprocess import clean_text


app = FastAPI(
    title="Spam Email Classifier API",
    description="API for detecting spam messages using Naive Bayes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ML components
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


class MessageRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "Spam Email Classifier API is running"}
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Naive Bayes"
    }

@app.post("/predict")
def predict(request: MessageRequest):

    # Check empty message
    if not request.message.strip():
        return {
            "error": "Message cannot be empty"
        }

    # Clean input
    cleaned_message = clean_text(request.message)

    # Convert to TF-IDF
    message_vector = vectorizer.transform([cleaned_message])

    # Prediction
    prediction = model.predict(message_vector)[0]

    # Probabilities
    probabilities = model.predict_proba(message_vector)[0]

    spam_probability = float(probabilities[1])

    # Confidence = probability of predicted class
    confidence = float(probabilities[prediction])

    return {
        "prediction": "SPAM" if prediction == 1 else "HAM",
        "spam_probability": round(spam_probability, 4),
        "confidence": round(confidence, 4)
    }