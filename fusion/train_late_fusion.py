from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

IMAGE_PREDICTIONS_DIR = PROJECT_ROOT / "image_branch" / "data" / "features" / "image_oof_stack" / "predictions"
TEXT_PREDICTIONS_DIR = PROJECT_ROOT / "text-branch" / "data" / "features" / "predictions"

IMAGE_TRAIN_PRED_PATH = IMAGE_PREDICTIONS_DIR / "image_oof_stacked_ensemble_train_predictions.csv"
IMAGE_VAL_PRED_PATH = IMAGE_PREDICTIONS_DIR / "image_oof_stacked_ensemble_val_predictions.csv"
IMAGE_TEST_PRED_PATH = IMAGE_PREDICTIONS_DIR / "image_oof_stacked_ensemble_test_predictions.csv"

TEXT_TRAIN_PRED_PATH = TEXT_PREDICTIONS_DIR / "text_train_predictions.csv"
TEXT_VAL_PRED_PATH = TEXT_PREDICTIONS_DIR / "text_val_predictions.csv"
TEXT_TEST_PRED_PATH = TEXT_PREDICTIONS_DIR / "text_test_predictions.csv"

FUSION_RESULTS_DIR = SCRIPT_DIR / "results"
FUSION_PREDICTIONS_DIR = SCRIPT_DIR / "predictions"

FUSION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FUSION_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

def clamp_rating_to_scale(predictions):
    return np.clip(predictions, 0.5, 5.0)


def regression_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def load_and_join_predictions(text_predictions_path, image_predictions_path, split_name):
    text_pred_df = pd.read_csv(text_predictions_path)
    image_pred_df = pd.read_csv(image_predictions_path)

    required_text = {"id", "y_true", "y_text_pred"}
    required_image = {"id", "y_true", "y_image_pred"}

    missing_text = required_text - set(text_pred_df.columns)
    missing_image = required_image - set(image_pred_df.columns)

    if missing_text:
        raise ValueError(f"{split_name} text file missing columns: {missing_text}")

    if missing_image:
        raise ValueError(f"{split_name} image file missing columns: {missing_image}")

    text_pred_df["id"] = text_pred_df["id"].astype(str)
    image_pred_df["id"] = image_pred_df["id"].astype(str)

    text_pred_df = text_pred_df.rename(columns={"y_true": "y_true_text"})
    image_pred_df = image_pred_df.rename(columns={"y_true": "y_true_image"})

    image_cols_to_keep = ["id", "y_true_image", "y_image_pred"]
    if "movie_title" in image_pred_df.columns:
        image_cols_to_keep.append("movie_title")

    merged_pred_df = text_pred_df.merge(
        image_pred_df[image_cols_to_keep],
        on="id",
        how="inner",
        suffixes=("_text", "_image")
    )

    if len(merged_pred_df) == 0:
        raise ValueError(f"{split_name}: merged 0 rows. Text/image ids do not match.")

    max_label_mismatch = np.max(np.abs(merged_pred_df["y_true_text"] - merged_pred_df["y_true_image"]))
    if max_label_mismatch > 1e-8:
        raise ValueError(f"{split_name}: y_true values do not match between text and image.")

    merged_pred_df = merged_pred_df.rename(columns={"y_true_text": "y_true"})
    merged_pred_df = merged_pred_df.drop(columns=["y_true_image"])
    if "movie_title_text" in merged_pred_df.columns:
        merged_pred_df = merged_pred_df.rename(columns={"movie_title_text": "movie_title"})
    elif "movie_title_image" in merged_pred_df.columns:
        merged_pred_df = merged_pred_df.rename(columns={"movie_title_image": "movie_title"})
    elif "movie_title" not in merged_pred_df.columns:
        merged_pred_df["movie_title"] = ""

    print(
        f"{split_name}: text rows={len(text_pred_df)}, image rows={len(image_pred_df)}, "
        f"merged rows={len(merged_pred_df)}"
    )

    return merged_pred_df


def build_result_row(model_name, input_name, y_val, val_pred, y_test, test_pred):
    val_metrics = regression_metrics(y_val, val_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    return {
        "model": model_name,
        "input": input_name,
        "val_mse": val_metrics["mse"],
        "val_mae": val_metrics["mae"],
        "val_r2": val_metrics["r2"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
    }

train_merged_df = load_and_join_predictions(TEXT_TRAIN_PRED_PATH, IMAGE_TRAIN_PRED_PATH, "train")
val_merged_df = load_and_join_predictions(TEXT_VAL_PRED_PATH, IMAGE_VAL_PRED_PATH, "val")
test_merged_df = load_and_join_predictions(TEXT_TEST_PRED_PATH, IMAGE_TEST_PRED_PATH, "test")

fusion_features_train = train_merged_df[["y_text_pred", "y_image_pred"]].values
fusion_features_val = val_merged_df[["y_text_pred", "y_image_pred"]].values
fusion_features_test = test_merged_df[["y_text_pred", "y_image_pred"]].values

ratings_train = train_merged_df["y_true"].astype(float).values
ratings_val = val_merged_df["y_true"].astype(float).values
ratings_test = test_merged_df["y_true"].astype(float).values

train_rating_mean = ratings_train.mean()

dummy_val_predictions = np.full_like(ratings_val, train_rating_mean, dtype=float)
dummy_test_predictions = np.full_like(ratings_test, train_rating_mean, dtype=float)

text_only_val_predictions = clamp_rating_to_scale(val_merged_df["y_text_pred"].values)
text_only_test_predictions = clamp_rating_to_scale(test_merged_df["y_text_pred"].values)

image_only_val_predictions = clamp_rating_to_scale(val_merged_df["y_image_pred"].values)
image_only_test_predictions = clamp_rating_to_scale(test_merged_df["y_image_pred"].values)

ridge_fusion_pipeline = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
)

ridge_fusion_pipeline.fit(fusion_features_train, ratings_train)

ridge_train_predictions = clamp_rating_to_scale(ridge_fusion_pipeline.predict(fusion_features_train))
ridge_val_predictions = clamp_rating_to_scale(ridge_fusion_pipeline.predict(fusion_features_val))
ridge_test_predictions = clamp_rating_to_scale(ridge_fusion_pipeline.predict(fusion_features_test))

linear_fusion_model = LinearRegression()
linear_fusion_model.fit(fusion_features_train, ratings_train)

linear_val_predictions = clamp_rating_to_scale(linear_fusion_model.predict(fusion_features_val))
linear_test_predictions = clamp_rating_to_scale(linear_fusion_model.predict(fusion_features_test))

best_text_weight = None
best_weighted_val_mse = float("inf")

for text_weight in np.linspace(0, 1, 101):
    candidate_val_predictions = clamp_rating_to_scale(
        text_weight * val_merged_df["y_text_pred"].values
        + (1 - text_weight) * val_merged_df["y_image_pred"].values
    )

    candidate_val_mse = mean_squared_error(ratings_val, candidate_val_predictions)

    if candidate_val_mse < best_weighted_val_mse:
        best_weighted_val_mse = candidate_val_mse
        best_text_weight = text_weight

weighted_val_predictions = clamp_rating_to_scale(
    best_text_weight * val_merged_df["y_text_pred"].values
    + (1 - best_text_weight) * val_merged_df["y_image_pred"].values
)

weighted_test_predictions = clamp_rating_to_scale(
    best_text_weight * test_merged_df["y_text_pred"].values
    + (1 - best_text_weight) * test_merged_df["y_image_pred"].values
)

hgb_fusion_model = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.03,
    max_leaf_nodes=15,
    l2_regularization=0.1,
    random_state=42
)

hgb_fusion_model.fit(fusion_features_train, ratings_train)

hgb_val_predictions = clamp_rating_to_scale(hgb_fusion_model.predict(fusion_features_val))
hgb_test_predictions = clamp_rating_to_scale(hgb_fusion_model.predict(fusion_features_test))

result_rows = []

result_rows.append(build_result_row(
    "Dummy Mean",
    "none",
    ratings_val,
    dummy_val_predictions,
    ratings_test,
    dummy_test_predictions
))

result_rows.append(build_result_row(
    "Text Only",
    "review text",
    ratings_val,
    text_only_val_predictions,
    ratings_test,
    text_only_test_predictions
))

result_rows.append(build_result_row(
    "Image Only OOF Stack",
    "poster image",
    ratings_val,
    image_only_val_predictions,
    ratings_test,
    image_only_test_predictions
))

result_rows.append(build_result_row(
    "Late Fusion RidgeCV",
    "text + image",
    ratings_val,
    ridge_val_predictions,
    ratings_test,
    ridge_test_predictions
))

result_rows.append(build_result_row(
    "Late Fusion LinearRegression",
    "text + image",
    ratings_val,
    linear_val_predictions,
    ratings_test,
    linear_test_predictions
))

result_rows.append(build_result_row(
    f"Weighted Average w_text={best_text_weight:.2f}",
    "text + image",
    ratings_val,
    weighted_val_predictions,
    ratings_test,
    weighted_test_predictions
))

result_rows.append(build_result_row(
    "Late Fusion HistGradientBoosting",
    "text + image",
    ratings_val,
    hgb_val_predictions,
    ratings_test,
    hgb_test_predictions
))

results_table = pd.DataFrame(result_rows).sort_values("val_mse")
results_path = FUSION_RESULTS_DIR / "fusion_results.csv"
results_table.to_csv(results_path, index=False)

print("\nFusion results sorted by validation MSE:")
print(results_table.round(4))

best_model_label = results_table.iloc[0]["model"]
if best_model_label == "Late Fusion RidgeCV":
    best_model_test_predictions = ridge_test_predictions
elif best_model_label == "Late Fusion LinearRegression":
    best_model_test_predictions = linear_test_predictions
elif best_model_label.startswith("Weighted Average"):
    best_model_test_predictions = weighted_test_predictions
elif best_model_label == "Late Fusion HistGradientBoosting":
    best_model_test_predictions = hgb_test_predictions
elif best_model_label == "Text Only":
    best_model_test_predictions = text_only_test_predictions
elif best_model_label == "Image Only OOF Stack":
    best_model_test_predictions = image_only_test_predictions
else:
    best_model_test_predictions = dummy_test_predictions

fusion_test_predictions_df = pd.DataFrame({
    "id": test_merged_df["id"].values,
    "movie_title": test_merged_df["movie_title"].values,
    "y_true": ratings_test,
    "y_text_pred": test_merged_df["y_text_pred"].values,
    "y_image_pred": test_merged_df["y_image_pred"].values,
    "y_ridge_fusion_pred": ridge_test_predictions,
    "y_linear_fusion_pred": linear_test_predictions,
    "y_weighted_fusion_pred": weighted_test_predictions,
    "y_hgb_fusion_pred": hgb_test_predictions,
    "y_best_pred": best_model_test_predictions,
})

pred_path = FUSION_PREDICTIONS_DIR / "fusion_test_predictions.csv"
fusion_test_predictions_df.to_csv(pred_path, index=False)

ridgecv_model = ridge_fusion_pipeline.named_steps["ridgecv"]

ridge_weight_table = pd.DataFrame({
    "feature": ["y_text_pred", "y_image_pred"],
    "ridge_weight_after_scaling": ridgecv_model.coef_,
})
ridge_weight_table["ridge_intercept"] = ridgecv_model.intercept_
ridge_weight_table.to_csv(FUSION_RESULTS_DIR / "fusion_ridge_weights.csv", index=False)

linear_weight_table = pd.DataFrame({
    "feature": ["y_text_pred", "y_image_pred"],
    "linear_weight": linear_fusion_model.coef_,
})
linear_weight_table["linear_intercept"] = linear_fusion_model.intercept_
linear_weight_table.to_csv(FUSION_RESULTS_DIR / "fusion_linear_weights.csv", index=False)

weighted_average_info_table = pd.DataFrame({
    "w_text": [best_text_weight],
    "w_image": [1 - best_text_weight],
    "val_mse": [best_weighted_val_mse],
})
weighted_average_info_table.to_csv(FUSION_RESULTS_DIR / "fusion_weighted_average_info.csv", index=False)


print("\nSaved:")
print(results_path)
print(pred_path)
print(FUSION_RESULTS_DIR / "fusion_ridge_weights.csv")
print(FUSION_RESULTS_DIR / "fusion_linear_weights.csv")
print(FUSION_RESULTS_DIR / "fusion_weighted_average_info.csv")

print("\nBest model by validation MSE:")
print(best_model_label)

print(f"\nBest weighted average: w_text={best_text_weight:.2f}, w_image={1 - best_text_weight:.2f}")