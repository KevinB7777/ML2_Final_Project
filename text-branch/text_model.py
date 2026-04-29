import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

IMAGE_PROCESSED_SPLITS_DIR = PROJECT_ROOT / "image_branch" / "data" / "processed"
TRAIN_SPLIT_PATH = IMAGE_PROCESSED_SPLITS_DIR / "train_with_posters_tvdb.json"
VAL_SPLIT_PATH = IMAGE_PROCESSED_SPLITS_DIR / "val_with_posters_tvdb.json"
TEST_SPLIT_PATH = IMAGE_PROCESSED_SPLITS_DIR / "test_with_posters_tvdb.json"

TEXT_FEATURES_DIR = SCRIPT_DIR / "data" / "features"
TEXT_PREDICTIONS_DIR = TEXT_FEATURES_DIR / "predictions"
TEXT_MODELS_DIR = TEXT_FEATURES_DIR / "models"

os.makedirs(TEXT_FEATURES_DIR, exist_ok=True)
os.makedirs(TEXT_PREDICTIONS_DIR, exist_ok=True)
os.makedirs(TEXT_MODELS_DIR, exist_ok=True)


def load_processed_split(split_path):
    split_df = pd.read_json(split_path)
    split_df["review_text"] = split_df["review_text"].fillna("")
    ratings = split_df["rating"].astype(float).values
    return split_df, ratings


def clamp_rating_to_scale(predictions):
    return np.clip(predictions, 0.5, 5.0)


def print_and_collect_metrics(y_true, y_pred, split_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{split_name}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2:  {r2:.4f}")

    return {
        "split": split_name,
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }


def write_predictions_csv(split_df, y_true, y_pred, out_path):
    pred_df = pd.DataFrame({
        "id": split_df["id"].astype(str).values,
        "movie_title": split_df["movie_title"].astype(str).values,
        "y_true": y_true,
        "y_text_pred": y_pred,
    })

    pred_df.to_csv(out_path, index=False)
    print("Saved:", out_path)


def main():
    print("Loading poster-filtered splits...")
    train_df, ratings_train = load_processed_split(TRAIN_SPLIT_PATH)
    val_df, ratings_val = load_processed_split(VAL_SPLIT_PATH)
    test_df, ratings_test = load_processed_split(TEST_SPLIT_PATH)

    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))
    print("Test rows:", len(test_df))

    print("\nFitting TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=20000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english"
    )

    X_train = tfidf_vectorizer.fit_transform(train_df["review_text"])
    X_val = tfidf_vectorizer.transform(val_df["review_text"])
    X_test = tfidf_vectorizer.transform(test_df["review_text"])

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    print("\nTraining RidgeCV text model...")
    ridge_alphas = np.logspace(-3, 3, 13)
    ridge_text_model = RidgeCV(alphas=ridge_alphas)
    ridge_text_model.fit(X_train, ratings_train)

    print("Selected alpha:", ridge_text_model.alpha_)

    train_predictions = clamp_rating_to_scale(ridge_text_model.predict(X_train))
    val_predictions = clamp_rating_to_scale(ridge_text_model.predict(X_val))
    test_predictions = clamp_rating_to_scale(ridge_text_model.predict(X_test))

    metric_rows = []
    metric_rows.append(print_and_collect_metrics(ratings_train, train_predictions, "train"))
    metric_rows.append(print_and_collect_metrics(ratings_val, val_predictions, "val"))
    metric_rows.append(print_and_collect_metrics(ratings_test, test_predictions, "test"))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = TEXT_FEATURES_DIR / "text_model_results.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("\nSaved:", metrics_path)

    write_predictions_csv(
        train_df,
        ratings_train,
        train_predictions,
        TEXT_PREDICTIONS_DIR / "text_train_predictions.csv"
    )

    write_predictions_csv(
        val_df,
        ratings_val,
        val_predictions,
        TEXT_PREDICTIONS_DIR / "text_val_predictions.csv"
    )

    write_predictions_csv(
        test_df,
        ratings_test,
        test_predictions,
        TEXT_PREDICTIONS_DIR / "text_test_predictions.csv"
    )

    joblib.dump(ridge_text_model, TEXT_MODELS_DIR / "ridge_model.pkl")
    joblib.dump(tfidf_vectorizer, TEXT_MODELS_DIR / "tfidf_vectorizer.pkl")

    print("\nSaved model/vectorizer.")
    print("Text branch is now fusion-ready.")


if __name__ == "__main__":
    main()