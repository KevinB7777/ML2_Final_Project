import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def process_text_features(data_path, is_train=True, vectorizer_path='text/tfidf_vectorizer.pkl'):
    """
    Converts text reviews into TF-IDF numerical features.
    """
    df = pd.read_json(data_path)
    df['review_text'] = df['review_text'].fillna("")
    y = df['rating'].values
    
    if is_train:
        # N-grams (1,3) to capture phrases like "absolute cinema"
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        X_text = vectorizer.fit_transform(df['review_text'])
        joblib.dump(vectorizer, vectorizer_path)
    else:
        # Load fitted vectorizer to prevent data leakage into val/test sets
        vectorizer = joblib.load(vectorizer_path)
        X_text = vectorizer.transform(df['review_text'])
        
    return X_text, y