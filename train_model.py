import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
from preprocess import preprocess_text

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset

    Args:
        filepath (str): Path to the CSV dataset

    Returns:
        tuple: (X, y) where X is preprocessed text and y is labels
    """
    # Load dataset
    df = pd.read_csv(filepath)

    # Preprocess text
    print("Preprocessing text data...")
    X = df['text'].apply(preprocess_text)
    y = df['sentiment']

    return X, y

def train_model(X, y):
    """
    Train the sentiment analysis model

    Args:
        X: Preprocessed text data
        y: Sentiment labels

    Returns:
        tuple: (model, vectorizer) trained model and TF-IDF vectorizer
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Logistic Regression classifier
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print results
    print("Model Training Complete!")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer, accuracy, precision, recall, f1

def save_model(model, vectorizer, accuracy, precision, recall, f1):
    """
    Save the trained model and vectorizer

    Args:
        model: Trained Logistic Regression model
        vectorizer: TF-IDF vectorizer
        accuracy, precision, recall, f1: Model metrics
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Save model and vectorizer
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    # Save evaluation results
    with open('results/evaluation.txt', 'w') as f:
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")

    print("\nModel saved to models/sentiment_model.pkl")
    print("Vectorizer saved to models/tfidf_vectorizer.pkl")
    print("Evaluation saved to results/evaluation.txt")

def main():
    """Main training function"""
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data('data/dataset.csv')

    print("Training model...")
    model, vectorizer, accuracy, precision, recall, f1 = train_model(X, y)

    print("Saving model...")
    save_model(model, vectorizer, accuracy, precision, recall, f1)

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()