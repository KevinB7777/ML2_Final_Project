import os
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


os.makedirs("fusion/results", exist_ok=True)
os.makedirs("fusion/predictions", exist_ok=True)


# --------------------------------------------------
# Paths
# --------------------------------------------------
IMAGE_TRAIN = "image_branch/data/features/image_oof_stack/predictions/image_oof_stacked_ensemble_train_predictions.csv"
IMAGE_VAL = "image_branch/data/features/image_oof_stack/predictions/image_oof_stacked_ensemble_val_predictions.csv"
IMAGE_TEST = "image_branch/data/features/image_oof_stack/predictions/image_oof_stacked_ensemble_test_predictions.csv"

TEXT_TRAIN = "text_branch/predictions/text_train_predictions.csv"
TEXT_VAL = "text_branch/predictions/text_val_predictions.csv"
TEXT_TEST = "text_branch/predictions/text_test_predictions.csv"


def clip_rating(pred):
    return np.clip(pred, 0.5, 5.0)


def metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def load_and_merge(text_path, image_path, split):
    text = pd.read_csv(text_path)
    image = pd.read_csv(image_path)

    required_text = {"id", "y_true", "y_text_pred"}
    required_image = {"id", "y_true", "y_image_pred"}

    missing_text = required_text - set(text.columns)
    missing_image = required_image - set(image.columns)

    if missing_text:
        raise ValueError(f"{split} text file missing columns: {missing_text}")

    if missing_image:
        raise ValueError(f"{split} image file missing columns: {missing_image}")

    # Rename y_true columns before merge to check consistency
    text = text.rename(columns={"y_true": "y_true_text"})
    image = image.rename(columns={"y_true": "y_true_image"})

    merged = text.merge(
        image[["id", "y_true_image", "y_image_pred"]],
        on="id",
        how="inner"
    )

    if len(merged) == 0:
        raise ValueError(f"{split}: merge produced 0 rows. IDs do not align.")

    # Check true labels match
    max_label_diff = np.max(np.abs(merged["y_true_text"] - merged["y_true_image"]))
    if max_label_diff > 1e-8:
        raise ValueError(f"{split}: y_true mismatch between text and image files.")

    merged = merged.rename(columns={"y_true_text": "y_true"})
    merged = merged.drop(columns=["y_true_image"])

    print(f"{split}: text rows={len(text)}, image rows={len(image)}, merged rows={len(merged)}")

    return merged


train = load_and_merge(TEXT_TRAIN, IMAGE_TRAIN, "train")
val = load_and_merge(TEXT_VAL, IMAGE_VAL, "val")
test = load_and_merge(TEXT_TEST, IMAGE_TEST, "test")


# --------------------------------------------------
# Prepare arrays
# --------------------------------------------------
X_train = train[["y_text_pred", "y_image_pred"]].values
X_val = val[["y_text_pred", "y_image_pred"]].values
X_test = test[["y_text_pred", "y_image_pred"]].values

y_train = train["y_true"].astype(float).values
y_val = val["y_true"].astype(float).values
y_test = test["y_true"].astype(float).values


# --------------------------------------------------
# Baselines
# --------------------------------------------------
dummy_train_pred = np.full_like(y_train, y_train.mean(), dtype=float)
dummy_val_pred = np.full_like(y_val, y_train.mean(), dtype=float)
dummy_test_pred = np.full_like(y_test, y_train.mean(), dtype=float)

text_val_pred = clip_rating(val["y_text_pred"].values)
text_test_pred = clip_rating(test["y_text_pred"].values)

image_val_pred = clip_rating(val["y_image_pred"].values)
image_test_pred = clip_rating(test["y_image_pred"].values)


# --------------------------------------------------
# Fusion model
# --------------------------------------------------
fusion_model = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
)

fusion_model.fit(X_train, y_train)

fusion_train_pred = clip_rating(fusion_model.predict(X_train))
fusion_val_pred = clip_rating(fusion_model.predict(X_val))
fusion_test_pred = clip_rating(fusion_model.predict(X_test))


# --------------------------------------------------
# Also test simple weighted average selected on val
# --------------------------------------------------
best_w = None
best_val_mse = float("inf")

for w_text in np.linspace(0, 1, 101):
    w_image = 1 - w_text
    pred = clip_rating(w_text * val["y_text_pred"].values + w_image * val["y_image_pred"].values)
    mse = mean_squared_error(y_val, pred)

    if mse < best_val_mse:
        best_val_mse = mse
        best_w = w_text

weighted_val_pred = clip_rating(best_w * val["y_text_pred"].values + (1 - best_w) * val["y_image_pred"].values)
weighted_test_pred = clip_rating(best_w * test["y_text_pred"].values + (1 - best_w) * test["y_image_pred"].values)


# --------------------------------------------------
# Results table
# --------------------------------------------------
rows = []

rows.append({
    "model": "Dummy Mean",
    "input": "none",
    "val_mse": metrics(y_val, dummy_val_pred)["mse"],
    "val_mae": metrics(y_val, dummy_val_pred)["mae"],
    "val_r2": metrics(y_val, dummy_val_pred)["r2"],
    "test_mse": metrics(y_test, dummy_test_pred)["mse"],
    "test_mae": metrics(y_test, dummy_test_pred)["mae"],
    "test_r2": metrics(y_test, dummy_test_pred)["r2"],
})

rows.append({
    "model": "Text Only",
    "input": "review text",
    "val_mse": metrics(y_val, text_val_pred)["mse"],
    "val_mae": metrics(y_val, text_val_pred)["mae"],
    "val_r2": metrics(y_val, text_val_pred)["r2"],
    "test_mse": metrics(y_test, text_test_pred)["mse"],
    "test_mae": metrics(y_test, text_test_pred)["mae"],
    "test_r2": metrics(y_test, text_test_pred)["r2"],
})

rows.append({
    "model": "Image Only OOF Stack",
    "input": "poster image",
    "val_mse": metrics(y_val, image_val_pred)["mse"],
    "val_mae": metrics(y_val, image_val_pred)["mae"],
    "val_r2": metrics(y_val, image_val_pred)["r2"],
    "test_mse": metrics(y_test, image_test_pred)["mse"],
    "test_mae": metrics(y_test, image_test_pred)["mae"],
    "test_r2": metrics(y_test, image_test_pred)["r2"],
})

rows.append({
    "model": "Late Fusion RidgeCV",
    "input": "text + image",
    "val_mse": metrics(y_val, fusion_val_pred)["mse"],
    "val_mae": metrics(y_val, fusion_val_pred)["mae"],
    "val_r2": metrics(y_val, fusion_val_pred)["r2"],
    "test_mse": metrics(y_test, fusion_test_pred)["mse"],
    "test_mae": metrics(y_test, fusion_test_pred)["mae"],
    "test_r2": metrics(y_test, fusion_test_pred)["r2"],
})

rows.append({
    "model": f"Weighted Average w_text={best_w:.2f}",
    "input": "text + image",
    "val_mse": metrics(y_val, weighted_val_pred)["mse"],
    "val_mae": metrics(y_val, weighted_val_pred)["mae"],
    "val_r2": metrics(y_val, weighted_val_pred)["r2"],
    "test_mse": metrics(y_test, weighted_test_pred)["mse"],
    "test_mae": metrics(y_test, weighted_test_pred)["mae"],
    "test_r2": metrics(y_test, weighted_test_pred)["r2"],
})

results = pd.DataFrame(rows).sort_values("val_mse")
results.to_csv("fusion/results/fusion_results.csv", index=False)

print("\nFusion results sorted by validation MSE:")
print(results.round(4))


# --------------------------------------------------
# Save fusion predictions
# --------------------------------------------------
fusion_pred_df = pd.DataFrame({
    "id": test["id"].values,
    "movie_title": test["movie_title"].values if "movie_title" in test.columns else "",
    "y_true": y_test,
    "y_text_pred": test["y_text_pred"].values,
    "y_image_pred": test["y_image_pred"].values,
    "y_fusion_pred": fusion_test_pred,
    "y_weighted_fusion_pred": weighted_test_pred,
})

fusion_pred_df.to_csv("fusion/predictions/fusion_test_predictions.csv", index=False)

# Save learned weights
ridge = fusion_model.named_steps["ridgecv"]

weights = pd.DataFrame({
    "feature": ["y_text_pred", "y_image_pred"],
    "weight": ridge.coef_,
})
weights["intercept"] = ridge.intercept_
weights.to_csv("fusion/results/fusion_ridge_weights.csv", index=False)

print("\nSaved:")
print("fusion/results/fusion_results.csv")
print("fusion/results/fusion_ridge_weights.csv")
print("fusion/predictions/fusion_test_predictions.csv")
print("\nFusion Ridge weights:")
print(weights)
print(f"\nBest weighted average used w_text={best_w:.2f}, w_image={1-best_w:.2f}")