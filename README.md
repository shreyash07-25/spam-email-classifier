# SpamGuard AI

A machine learning-based spam email classifier that detects whether a message is **SPAM** or **HAM** using TF-IDF feature extraction and Multinomial Naive Bayes.

🔗 **Live Demo:** https://spam-email-classifier-18jhikhkk-just-bc32.vercel.app/

## Features

* Spam/HAM email classification
* TF-IDF text feature extraction
* Multinomial Naive Bayes classification
* Confidence score for predictions
* FastAPI REST API
* Responsive web interface
* Real-time prediction
* Deployed frontend and backend

## Tech Stack

### Machine Learning

* Python
* Scikit-learn
* Pandas
* TF-IDF
* Multinomial Naive Bayes
* Joblib

### Backend

* FastAPI
* Uvicorn
* REST API

### Frontend

* HTML
* JavaScript
* Tailwind CSS

### Deployment

* Vercel — Frontend
* Render — Backend

## Project Structure

```text
spam-email-classifier/
│
├── data/
│   └── spam.csv
│
├── ml/
│   ├── preprocess.py
│   ├── check_data.py
│   ├── train.py
│   ├── compare_models.py
│   └── predict.py
│
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── api/
│   └── main.py
│
├── frontend/
│   ├── index.html
│   └── script.js
│
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

## How It Works

The application follows this pipeline:

```text
User Message
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Multinomial Naive Bayes
     ↓
SPAM / HAM Prediction
     ↓
Confidence Score
```

## Machine Learning

The model uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text messages into numerical feature vectors.

The classifier is a **Multinomial Naive Bayes** model trained on a labeled spam/ham dataset containing more than 5,000 messages.

The training pipeline includes:

1. Loading the dataset
2. Removing unnecessary columns
3. Cleaning text
4. Splitting data into training and testing sets
5. TF-IDF feature extraction
6. Training the Naive Bayes classifier
7. Evaluating the model
8. Saving the trained model and vectorizer

## API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "Naive Bayes"
}
```

### Spam Prediction

```http
POST /predict
```

Request:

```json
{
  "message": "Congratulations! You have won a free cash prize!"
}
```

Response:

```json
{
  "prediction": "SPAM",
  "spam_probability": 0.98,
  "confidence": 0.98
}
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/shreyash07-25/spam-email-classifier.git
cd spam-email-classifier
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python ml/train.py
```

This generates:

```text
models/spam_model.pkl
models/tfidf_vectorizer.pkl
```

### 5. Start the FastAPI server

```bash
python -m uvicorn api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Run the frontend

Open:

```text
frontend/index.html
```

in your browser.

For local development, update the API URL in `frontend/script.js`:

```javascript
const API_URL = "http://127.0.0.1:8000";
```

## Deployment

The application is deployed using a separate frontend and backend architecture:

```text
                 ┌──────────────────┐
                 │   Vercel          │
                 │   Frontend       │
                 └────────┬─────────┘
                          │
                          │ REST API
                          ↓
                 ┌──────────────────┐
                 │   Render         │
                 │   FastAPI        │
                 └────────┬─────────┘
                          │
                          ↓
                 ┌──────────────────┐
                 │ ML Model         │
                 │ TF-IDF + NB      │
                 └──────────────────┘
```

## Future Improvements

* Improve spam recall with better preprocessing and model tuning
* Experiment with Logistic Regression and other ML algorithms
* Add a larger and more diverse dataset
* Add explainable predictions
* Experiment with transformer-based NLP models
* Add user authentication and prediction history

## Author

**Shreyash Sinha**

GitHub: https://github.com/shreyash07-25
