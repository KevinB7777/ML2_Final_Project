import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TFIDF_VECTORIZER_PATH = SCRIPT_DIR / "data" / "features" / "models" / "tfidf_vectorizer.pkl"


def process_text_features(split_path, fit_vectorizer=True, tfidf_vectorizer_path=None):
    if tfidf_vectorizer_path is None:
        tfidf_vectorizer_path = DEFAULT_TFIDF_VECTORIZER_PATH

    tfidf_vectorizer_path = Path(tfidf_vectorizer_path)
    split_df = pd.read_json(split_path)
    split_df['review_text'] = split_df['review_text'].fillna("")
    ratings = split_df['rating'].values
    
    if fit_vectorizer:
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        tfidf_features = tfidf_vectorizer.fit_transform(split_df['review_text'])
        tfidf_vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
    else:
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        tfidf_features = tfidf_vectorizer.transform(split_df['review_text'])
        
    return tfidf_features, ratings