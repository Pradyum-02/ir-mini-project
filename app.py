
from pathlib import Path

import joblib
from flask import Flask, render_template, request, jsonify
from preprocess import preprocess_text

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

model = None
vectorizer = None


def load_model():
    """Load the trained classifier and TF-IDF vectorizer."""
    global model, vectorizer

    if model is not None and vectorizer is not None:
        return True

    try:
        if not MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file():
            app.logger.error("Trained model or vectorizer file is missing.")
            return False

        loaded_model = joblib.load(MODEL_PATH)
        loaded_vectorizer = joblib.load(VECTORIZER_PATH)

        model = loaded_model
        vectorizer = loaded_vectorizer

        app.logger.info("Sentiment model and vectorizer loaded.")
        return True

    except Exception:
        app.logger.exception("Failed to load sentiment model.")
        return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not load_model():
        return jsonify({
            "error": "Model unavailable. Check the deployed model files and logs."
        }), 500

    try:
        text = request.form.get("text", "")

        if not text.strip():
            return jsonify({"error": "Please enter some text"}), 400

        processed_text = preprocess_text(text)
        text_tfidf = vectorizer.transform([processed_text])

        prediction = str(model.predict(text_tfidf)[0]).lower()
        probabilities = model.predict_proba(text_tfidf)[0]

        confidence = float(max(probabilities)) * 100

        sentiment_map = {
            "positive": ("POSITIVE", "😊"),
            "negative": ("NEGATIVE", "😡"),
            "neutral": ("NEUTRAL", "😐"),
        }

        sentiment_label, emoji = sentiment_map.get(
            prediction, ("UNKNOWN", "❓")
        )

        return jsonify({
            "sentiment": sentiment_label,
            "emoji": emoji,
            "confidence": round(confidence, 2),
            "original_text": text,
        })

    except Exception:
        app.logger.exception("Prediction failed.")
        return jsonify({
            "error": "Prediction failed. Check the application logs."
        }), 500


if __name__ == "__main__":
    if not load_model():
        app.logger.warning("Starting without loaded models.")

    app.run(debug=False, host="0.0.0.0", port=5000)
