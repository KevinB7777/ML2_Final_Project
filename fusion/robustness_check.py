from pathlib import Path
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

IMAGE_PREDICTIONS_DIR = PROJECT_ROOT / "image_branch" / "data" / "features" / "image_oof_stack" / "predictions"
TEXT_PREDICTIONS_DIR = PROJECT_ROOT / "text-branch" / "data" / "features" / "predictions"

IMAGE_TRAIN = IMAGE_PREDICTIONS_DIR / "image_oof_stacked_ensemble_train_predictions.csv"
IMAGE_VAL = IMAGE_PREDICTIONS_DIR / "image_oof_stacked_ensemble_val_predictions.csv"
IMAGE_TEST = IMAGE_PREDICTIONS_DIR / "image_oof_stacked_ensemble_test_predictions.csv"

TEXT_TRAIN = TEXT_PREDICTIONS_DIR / "text_train_predictions.csv"
TEXT_VAL = TEXT_PREDICTIONS_DIR / "text_val_predictions.csv"
TEXT_TEST = TEXT_PREDICTIONS_DIR / "text_test_predictions.csv"

OUT_DIR = SCRIPT_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clamp(pred):
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

    text["id"] = text["id"].astype(str)
    image["id"] = image["id"].astype(str)

    text = text.rename(columns={"y_true": "y_true_text"})
    image = image.rename(columns={"y_true": "y_true_image"})

    merged = text.merge(
        image[["id", "y_true_image", "y_image_pred"]],
        on="id",
        how="inner"
    )

    if len(merged) == 0:
        raise ValueError(f"{split}: merge produced 0 rows.")

    max_diff = np.max(np.abs(merged["y_true_text"] - merged["y_true_image"]))
    if max_diff > 1e-8:
        raise ValueError(f"{split}: y_true mismatch between text and image.")

    merged = merged.rename(columns={"y_true_text": "y_true"})
    merged = merged.drop(columns=["y_true_image"])

    print(f"{split}: merged rows={len(merged)}")
    return merged


train_df = load_and_merge(TEXT_TRAIN, IMAGE_TRAIN, "train")
val_df = load_and_merge(TEXT_VAL, IMAGE_VAL, "val")
test_df = load_and_merge(TEXT_TEST, IMAGE_TEST, "test")

X_train = train_df[["y_text_pred", "y_image_pred"]].values
X_val = val_df[["y_text_pred", "y_image_pred"]].values
X_test = test_df[["y_text_pred", "y_image_pred"]].values

y_train = train_df["y_true"].astype(float).values
y_val = val_df["y_true"].astype(float).values
y_test = test_df["y_true"].astype(float).values

text_val_pred = clamp(val_df["y_text_pred"].values)
text_test_pred = clamp(test_df["y_text_pred"].values)

image_val_pred = clamp(val_df["y_image_pred"].values)
image_test_pred = clamp(test_df["y_image_pred"].values)

dummy_val_pred = np.full_like(y_val, y_train.mean(), dtype=float)
dummy_test_pred = np.full_like(y_test, y_train.mean(), dtype=float)

text_val_m = metrics(y_val, text_val_pred)
text_test_m = metrics(y_test, text_test_pred)
image_test_m = metrics(y_test, image_test_pred)
dummy_test_m = metrics(y_test, dummy_test_pred)

print("\nFixed baselines:")
print("Dummy Test:", dummy_test_m)
print("Image Test:", image_test_m)
print("Text  Test:", text_test_m)


def run_hgb_trial(seed):
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.03,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=seed
    )

    model.fit(X_train, y_train)

    val_pred = clamp(model.predict(X_val))
    test_pred = clamp(model.predict(X_test))

    val_m = metrics(y_val, val_pred)
    test_m = metrics(y_test, test_pred)

    return {
        "seed": seed,
        "hgb_val_mse": val_m["mse"],
        "hgb_val_mae": val_m["mae"],
        "hgb_val_r2": val_m["r2"],
        "hgb_test_mse": test_m["mse"],
        "hgb_test_mae": test_m["mae"],
        "hgb_test_r2": test_m["r2"],
        "text_test_mse": text_test_m["mse"],
        "text_test_mae": text_test_m["mae"],
        "text_test_r2": text_test_m["r2"],
        "test_mse_improvement": text_test_m["mse"] - test_m["mse"],
        "test_mae_improvement": text_test_m["mae"] - test_m["mae"],
        "beats_text_mse": test_m["mse"] < text_test_m["mse"],
        "beats_text_mae": test_m["mae"] < text_test_m["mae"],
    }


seeds = list(range(10))

# n_jobs=-1 uses all CPU cores. If your laptop gets too hot, change to n_jobs=4.
results = Parallel(n_jobs=-1)(
    delayed(run_hgb_trial)(seed) for seed in seeds
)

results_df = pd.DataFrame(results)
results_path = OUT_DIR / "fusion_hgb_robustness_10_trials.csv"
results_df.to_csv(results_path, index=False)

summary = {
    "num_trials": len(results_df),
    "text_test_mse": text_test_m["mse"],
    "text_test_mae": text_test_m["mae"],
    "text_test_r2": text_test_m["r2"],
    "hgb_test_mse_mean": results_df["hgb_test_mse"].mean(),
    "hgb_test_mse_std": results_df["hgb_test_mse"].std(),
    "hgb_test_mse_min": results_df["hgb_test_mse"].min(),
    "hgb_test_mse_max": results_df["hgb_test_mse"].max(),
    "hgb_test_mae_mean": results_df["hgb_test_mae"].mean(),
    "hgb_test_mae_std": results_df["hgb_test_mae"].std(),
    "hgb_test_r2_mean": results_df["hgb_test_r2"].mean(),
    "hgb_test_r2_std": results_df["hgb_test_r2"].std(),
    "mean_mse_improvement_over_text": results_df["test_mse_improvement"].mean(),
    "std_mse_improvement_over_text": results_df["test_mse_improvement"].std(),
    "min_mse_improvement_over_text": results_df["test_mse_improvement"].min(),
    "max_mse_improvement_over_text": results_df["test_mse_improvement"].max(),
    "fusion_beats_text_mse_count": int(results_df["beats_text_mse"].sum()),
    "fusion_beats_text_mse_rate": float(results_df["beats_text_mse"].mean()),
}

summary_df = pd.DataFrame([summary])
summary_path = OUT_DIR / "fusion_hgb_robustness_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nRobustness results:")
print(results_df.round(4))

print("\nSummary:")
print(summary_df.round(4).T)

print("\nSaved:")
print(results_path)
print(summary_path)