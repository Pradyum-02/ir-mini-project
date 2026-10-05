import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data packages"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

def preprocess_text(text):
    """
    Preprocess text for sentiment analysis

    Args:
        text (str): Input text to preprocess

    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags (but keep the text)
    text = re.sub(r'#', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Join tokens back to text
    return ' '.join(tokens)

if __name__ == "__main__":
    # Test the preprocessing function
    test_texts = [
        "I love this new phone! It's amazing!",
        "This is the worst experience ever.",
        "The weather is okay today."
    ]

    for text in test_texts:
        print(f"Original: {text}")
        print(f"Preprocessed: {preprocess_text(text)}")
        print("-" * 50)