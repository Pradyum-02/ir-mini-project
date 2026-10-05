# Tweet Sentiment Analysis System

A simple NLP-based sentiment analysis system for classifying tweets as Positive, Negative, or Neutral.

## Features

- Text preprocessing (lowercase, URL removal, @mention removal, hashtag handling, etc.)
- TF-IDF feature extraction
- Logistic Regression classification
- Clean web interface built with Flask
- Real-time sentiment prediction
- Example tweets for quick testing
- Model performance metrics display

## Project Structure

```
tweet-sentiment-analysis/
│
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── preprocess.py          # Text preprocessing functions
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── data/
│   └── dataset.csv        # Training dataset
│
├── models/
│   ├── sentiment_model.pkl        # Trained classifier
│   └── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer
│
├── templates/
│   └── index.html         # Main HTML template
│
├── static/
│   ├── style.css          # Stylesheet
│   └── script.js          # Client-side JavaScript
│
└── results/
    └── evaluation.txt     # Model performance metrics
```

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the model:
   ```bash
   python train_model.py
   ```

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and visit: http://localhost:5000

## How It Works

### Text Preprocessing
The system preprocesses input text by:
- Converting to lowercase
- Removing URLs, @mentions, and hashtags
- Removing special characters and extra whitespace
- Removing stopwords
- Applying stemming

### Machine Learning Pipeline
1. Convert preprocessed text to TF-IDF features
2. Train a Logistic Regression classifier
3. Evaluate model performance (accuracy, precision, recall, F1-score)
4. Save the trained model and vectorizer

### Web Interface
The Flask app provides a clean interface where users can:
- Enter text for sentiment analysis
- Get real-time predictions with confidence scores
- Click example tweets to test the system
- View model information and performance metrics

## Model Information

- **Algorithm**: Logistic Regression
- **Features**: TF-IDF ( unigrams and bigrams )
- **Classes**: Positive, Negative, Neutral
- **Input**: Preprocessed text tweets
- **Output**: Sentiment label with confidence percentage

## Example Usage

Input: "I absolutely love this new phone!"
Output: Positive 😊 (Confidence: 85.2%)

Input: "This is the worst service ever."
Output: Negative 😡 (Confidence: 92.1%)

Input: "The weather is okay today."
Output: Neutral 😐 (Confidence: 78.5%)

## Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- nltk
- joblib

## Notes

This is a educational project designed for simplicity and clarity. For production use, consider:
- Using larger, more diverse datasets
- Implementing more sophisticated preprocessing
- Trying different algorithms (Naive Bayes, SVM, etc.)
- Adding cross-validation and hyperparameter tuning
- Implementing proper error handling and logging