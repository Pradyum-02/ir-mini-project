
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Store NLTK resources in a writable temporary directory on Vercel.
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_DIR)


def download_nltk_data():
    """Download required NLTK resources if they are missing."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
    }

    for resource_path, package in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            success = nltk.download(
                package,
                download_dir=NLTK_DATA_DIR,
                quiet=True,
            )
            if not success:
                raise RuntimeError(
                    f"Could not download required NLTK resource: {package}"
                )


download_nltk_data()


def preprocess_text(text):
    """
    Preprocess text for sentiment analysis.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r'http\S+|www\S+|https\S+',
        '',
        text,
        flags=re.MULTILINE,
    )

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
    test_texts = [
        "I love this new phone! It's amazing!",
        "This is the worst experience ever.",
        "The weather is okay today.",
    ]

    for text in test_texts:
        print(f"Original: {text}")
        print(f"Preprocessed: {preprocess_text(text)}")
        print("-" * 50)
