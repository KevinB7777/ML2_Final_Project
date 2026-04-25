import os
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


TRAIN_PATH = "image_branch/data/processed/train_with_posters_tvdb.json"
VAL_PATH = "image_branch/data/processed/val_with_posters_tvdb.json"
TEST_PATH = "image_branch/data/processed/test_with_posters_tvdb.json"

os.makedirs("text", exist_ok=True)
os.makedirs("fusion", exist_ok=True)
os.makedirs("text/predictions", exist_ok=True)


def load_split(path):
    df = pd.read_json(path)
    df["review_text"] = df["review_text"].fillna("")
    y = df["rating"].astype(float).values
    return df, y


def clip_rating(pred):
    return np.clip(pred, 0.5, 5.0)


def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2:  {r2:.4f}")

    return {
        "split": name,
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }


def save_predictions(df, y_true, y_pred, out_path):
    pred_df = pd.DataFrame({
        "id": df["id"].astype(str).values,
        "movie_title": df["movie_title"].astype(str).values,
        "y_true": y_true,
        "y_text_pred": y_pred,
    })

    pred_df.to_csv(out_path, index=False)
    print("Saved:", out_path)


def main():
    print("Loading poster-filtered splits...")
    train_df, y_train = load_split(TRAIN_PATH)
    val_df, y_val = load_split(VAL_PATH)
    test_df, y_test = load_split(TEST_PATH)

    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))
    print("Test rows:", len(test_df))

    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=20000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(train_df["review_text"])
    X_val = vectorizer.transform(val_df["review_text"])
    X_test = vectorizer.transform(test_df["review_text"])

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    print("\nTraining RidgeCV text model...")
    alphas = np.logspace(-3, 3, 13)
    model = RidgeCV(alphas=alphas)
    model.fit(X_train, y_train)

    print("Selected alpha:", model.alpha_)

    train_pred = clip_rating(model.predict(X_train))
    val_pred = clip_rating(model.predict(X_val))
    test_pred = clip_rating(model.predict(X_test))

    results = []
    results.append(evaluate(y_train, train_pred, "train"))
    results.append(evaluate(y_val, val_pred, "val"))
    results.append(evaluate(y_test, test_pred, "test"))

    results_df = pd.DataFrame(results)
    results_df.to_csv("text/text_model_results.csv", index=False)
    print("\nSaved: text/text_model_results.csv")

    save_predictions(
        train_df,
        y_train,
        train_pred,
        "text/predictions/text_train_predictions.csv"
    )

    save_predictions(
        val_df,
        y_val,
        val_pred,
        "text/predictions/text_val_predictions.csv"
    )

    save_predictions(
        test_df,
        y_test,
        test_pred,
        "text/predictions/text_test_predictions.csv"
    )

    joblib.dump(model, "text/ridge_model.pkl")
    joblib.dump(vectorizer, "text/tfidf_vectorizer.pkl")

    print("\nSaved model/vectorizer.")
    print("Text branch is now fusion-ready.")


if __name__ == "__main__":
    main()