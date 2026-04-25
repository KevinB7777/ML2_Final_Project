import os
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# --------------------------------------------------
# Paths
# --------------------------------------------------
IMAGE_TRAIN = "image_branch/data/features/image_oof_stack/predictions/image_oof_stacked_ensemble_train_predictions.csv"
IMAGE_VAL = "image_branch/data/features/image_oof_stack/predictions/image_oof_stacked_ensemble_val_predictions.csv"
IMAGE_TEST = "image_branch/data/features/image_oof_stack/predictions/image_oof_stacked_ensemble_test_predictions.csv"

TEXT_TRAIN = "ml-project/text/predictions/text_train_predictions.csv"
TEXT_VAL = "ml-project/text/predictions/text_val_predictions.csv"
TEXT_TEST = "ml-project/text/predictions/text_test_predictions.csv"

OUT_DIR = "fusion"
RESULTS_DIR = os.path.join(OUT_DIR, "results")
PRED_DIR = os.path.join(OUT_DIR, "predictions")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def clip_rating(pred):
    return np.clip(pred, 0.5, 5.0)


def metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def load_and_merge(text_path, image_path, split_name):
    text = pd.read_csv(text_path)
    image = pd.read_csv(image_path)

    required_text = {"id", "y_true", "y_text_pred"}
    required_image = {"id", "y_true", "y_image_pred"}

    missing_text = required_text - set(text.columns)
    missing_image = required_image - set(image.columns)

    if missing_text:
        raise ValueError(f"{split_name} text file missing columns: {missing_text}")

    if missing_image:
        raise ValueError(f"{split_name} image file missing columns: {missing_image}")

    text["id"] = text["id"].astype(str)
    image["id"] = image["id"].astype(str)

    text = text.rename(columns={"y_true": "y_true_text"})
    image = image.rename(columns={"y_true": "y_true_image"})

    keep_image_cols = ["id", "y_true_image", "y_image_pred"]
    if "movie_title" in image.columns:
        keep_image_cols.append("movie_title")

    merged = text.merge(
        image[keep_image_cols],
        on="id",
        how="inner",
        suffixes=("_text", "_image")
    )

    if len(merged) == 0:
        raise ValueError(f"{split_name}: merged 0 rows. Text/image ids do not match.")

    max_label_diff = np.max(np.abs(merged["y_true_text"] - merged["y_true_image"]))
    if max_label_diff > 1e-8:
        raise ValueError(f"{split_name}: y_true values do not match between text and image.")

    merged = merged.rename(columns={"y_true_text": "y_true"})
    merged = merged.drop(columns=["y_true_image"])

    # Clean movie_title column after merge
    if "movie_title_text" in merged.columns:
        merged = merged.rename(columns={"movie_title_text": "movie_title"})
    elif "movie_title_image" in merged.columns:
        merged = merged.rename(columns={"movie_title_image": "movie_title"})
    elif "movie_title" not in merged.columns:
        merged["movie_title"] = ""

    print(
        f"{split_name}: text rows={len(text)}, image rows={len(image)}, "
        f"merged rows={len(merged)}"
    )

    return merged


def evaluate_row(model_name, input_name, y_val, val_pred, y_test, test_pred):
    val_m = metrics(y_val, val_pred)
    test_m = metrics(y_test, test_pred)

    return {
        "model": model_name,
        "input": input_name,
        "val_mse": val_m["mse"],
        "val_mae": val_m["mae"],
        "val_r2": val_m["r2"],
        "test_mse": test_m["mse"],
        "test_mae": test_m["mae"],
        "test_r2": test_m["r2"],
    }


# --------------------------------------------------
# Load merged predictions
# --------------------------------------------------
train = load_and_merge(TEXT_TRAIN, IMAGE_TRAIN, "train")
val = load_and_merge(TEXT_VAL, IMAGE_VAL, "val")
test = load_and_merge(TEXT_TEST, IMAGE_TEST, "test")

X_train = train[["y_text_pred", "y_image_pred"]].values
X_val = val[["y_text_pred", "y_image_pred"]].values
X_test = test[["y_text_pred", "y_image_pred"]].values

y_train = train["y_true"].astype(float).values
y_val = val["y_true"].astype(float).values
y_test = test["y_true"].astype(float).values


# --------------------------------------------------
# Baselines
# --------------------------------------------------
train_mean = y_train.mean()

dummy_val_pred = np.full_like(y_val, train_mean, dtype=float)
dummy_test_pred = np.full_like(y_test, train_mean, dtype=float)

text_val_pred = clip_rating(val["y_text_pred"].values)
text_test_pred = clip_rating(test["y_text_pred"].values)

image_val_pred = clip_rating(val["y_image_pred"].values)
image_test_pred = clip_rating(test["y_image_pred"].values)


# --------------------------------------------------
# Fusion Model 1: RidgeCV late fusion
# --------------------------------------------------
ridge_fusion = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
)

ridge_fusion.fit(X_train, y_train)

ridge_train_pred = clip_rating(ridge_fusion.predict(X_train))
ridge_val_pred = clip_rating(ridge_fusion.predict(X_val))
ridge_test_pred = clip_rating(ridge_fusion.predict(X_test))


# --------------------------------------------------
# Fusion Model 2: Linear regression late fusion
# --------------------------------------------------
linear_fusion = LinearRegression()
linear_fusion.fit(X_train, y_train)

linear_val_pred = clip_rating(linear_fusion.predict(X_val))
linear_test_pred = clip_rating(linear_fusion.predict(X_test))


# --------------------------------------------------
# Fusion Model 3: Weighted average selected on validation
# final = w * text + (1-w) * image
# --------------------------------------------------
best_w = None
best_val_mse = float("inf")

for w_text in np.linspace(0, 1, 101):
    pred = clip_rating(
        w_text * val["y_text_pred"].values
        + (1 - w_text) * val["y_image_pred"].values
    )

    mse = mean_squared_error(y_val, pred)

    if mse < best_val_mse:
        best_val_mse = mse
        best_w = w_text

weighted_val_pred = clip_rating(
    best_w * val["y_text_pred"].values
    + (1 - best_w) * val["y_image_pred"].values
)

weighted_test_pred = clip_rating(
    best_w * test["y_text_pred"].values
    + (1 - best_w) * test["y_image_pred"].values
)


# --------------------------------------------------
# Fusion Model 4: Small nonlinear fusion
# Only 2 inputs, so keep it simple.
# --------------------------------------------------
hgb_fusion = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.03,
    max_leaf_nodes=15,
    l2_regularization=0.1,
    random_state=42
)

hgb_fusion.fit(X_train, y_train)

hgb_val_pred = clip_rating(hgb_fusion.predict(X_val))
hgb_test_pred = clip_rating(hgb_fusion.predict(X_test))


# --------------------------------------------------
# Results table
# --------------------------------------------------
rows = []

rows.append(evaluate_row(
    "Dummy Mean",
    "none",
    y_val,
    dummy_val_pred,
    y_test,
    dummy_test_pred
))

rows.append(evaluate_row(
    "Text Only",
    "review text",
    y_val,
    text_val_pred,
    y_test,
    text_test_pred
))

rows.append(evaluate_row(
    "Image Only OOF Stack",
    "poster image",
    y_val,
    image_val_pred,
    y_test,
    image_test_pred
))

rows.append(evaluate_row(
    "Late Fusion RidgeCV",
    "text + image",
    y_val,
    ridge_val_pred,
    y_test,
    ridge_test_pred
))

rows.append(evaluate_row(
    "Late Fusion LinearRegression",
    "text + image",
    y_val,
    linear_val_pred,
    y_test,
    linear_test_pred
))

rows.append(evaluate_row(
    f"Weighted Average w_text={best_w:.2f}",
    "text + image",
    y_val,
    weighted_val_pred,
    y_test,
    weighted_test_pred
))

rows.append(evaluate_row(
    "Late Fusion HistGradientBoosting",
    "text + image",
    y_val,
    hgb_val_pred,
    y_test,
    hgb_test_pred
))

results = pd.DataFrame(rows).sort_values("val_mse")
results_path = os.path.join(RESULTS_DIR, "fusion_results.csv")
results.to_csv(results_path, index=False)

print("\nFusion results sorted by validation MSE:")
print(results.round(4))


# --------------------------------------------------
# Save best fusion test predictions
# --------------------------------------------------
best_model_name = results.iloc[0]["model"]

# choose prediction array corresponding to best model
if best_model_name == "Late Fusion RidgeCV":
    best_test_pred = ridge_test_pred
elif best_model_name == "Late Fusion LinearRegression":
    best_test_pred = linear_test_pred
elif best_model_name.startswith("Weighted Average"):
    best_test_pred = weighted_test_pred
elif best_model_name == "Late Fusion HistGradientBoosting":
    best_test_pred = hgb_test_pred
elif best_model_name == "Text Only":
    best_test_pred = text_test_pred
elif best_model_name == "Image Only OOF Stack":
    best_test_pred = image_test_pred
else:
    best_test_pred = dummy_test_pred

test_pred_df = pd.DataFrame({
    "id": test["id"].values,
    "movie_title": test["movie_title"].values,
    "y_true": y_test,
    "y_text_pred": test["y_text_pred"].values,
    "y_image_pred": test["y_image_pred"].values,
    "y_ridge_fusion_pred": ridge_test_pred,
    "y_linear_fusion_pred": linear_test_pred,
    "y_weighted_fusion_pred": weighted_test_pred,
    "y_hgb_fusion_pred": hgb_test_pred,
    "y_best_pred": best_test_pred,
})

pred_path = os.path.join(PRED_DIR, "fusion_test_predictions.csv")
test_pred_df.to_csv(pred_path, index=False)


# --------------------------------------------------
# Save fusion weights
# --------------------------------------------------
ridge_model = ridge_fusion.named_steps["ridgecv"]

ridge_weights = pd.DataFrame({
    "feature": ["y_text_pred", "y_image_pred"],
    "ridge_weight_after_scaling": ridge_model.coef_,
})
ridge_weights["ridge_intercept"] = ridge_model.intercept_
ridge_weights.to_csv(os.path.join(RESULTS_DIR, "fusion_ridge_weights.csv"), index=False)

linear_weights = pd.DataFrame({
    "feature": ["y_text_pred", "y_image_pred"],
    "linear_weight": linear_fusion.coef_,
})
linear_weights["linear_intercept"] = linear_fusion.intercept_
linear_weights.to_csv(os.path.join(RESULTS_DIR, "fusion_linear_weights.csv"), index=False)

weighted_info = pd.DataFrame({
    "w_text": [best_w],
    "w_image": [1 - best_w],
    "val_mse": [best_val_mse],
})
weighted_info.to_csv(os.path.join(RESULTS_DIR, "fusion_weighted_average_info.csv"), index=False)


print("\nSaved:")
print(results_path)
print(pred_path)
print(os.path.join(RESULTS_DIR, "fusion_ridge_weights.csv"))
print(os.path.join(RESULTS_DIR, "fusion_linear_weights.csv"))
print(os.path.join(RESULTS_DIR, "fusion_weighted_average_info.csv"))

print("\nBest model by validation MSE:")
print(best_model_name)

print(f"\nBest weighted average: w_text={best_w:.2f}, w_image={1 - best_w:.2f}")