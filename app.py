from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from preprocess import preprocess_text

app = Flask(__name__)

# Load model and vectorizer
model = None
vectorizer = None

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    try:
        model_path = 'models/sentiment_model.pkl'
        vectorizer_path = 'models/tfidf_vectorizer.pkl'

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            print("Model and vectorizer loaded successfully!")
            return True
        else:
            print("Model files not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle sentiment prediction"""
    if model is None or vectorizer is None:
        if not load_model():
            return jsonify({'error': 'Model not available. Please train the model first.'}), 500

    try:
        # Get text from form
        text = request.form.get('text', '')

        if not text.strip():
            return jsonify({'error': 'Please enter some text'}), 400

        # Preprocess the text
        processed_text = preprocess_text(text)

        # Vectorize the text
        text_tfidf = vectorizer.transform([processed_text])

        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]

        # Get confidence score
        confidence = max(probabilities) * 100

        # Map prediction to sentiment with emoji
        sentiment_map = {
            'positive': ('POSITIVE', '😊'),
            'negative': ('NEGATIVE', '😡'),
            'neutral': ('NEUTRAL', '😐')
        }

        sentiment_label, emoji = sentiment_map.get(prediction, ('UNKNOWN', '❓'))

        return jsonify({
            'sentiment': sentiment_label,
            'emoji': emoji,
            'confidence': round(confidence, 2),
            'original_text': text
        })

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    print("Starting Flask application...")
    print("Visit http://localhost:5000 to use the sentiment analyzer")
    app.run(debug=True, host='0.0.0.0', port=5000)