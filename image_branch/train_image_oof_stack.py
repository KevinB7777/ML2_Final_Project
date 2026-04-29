from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
import joblib

warnings.filterwarnings("ignore")

N_FOLDS = 5
RANDOM_STATE = 42

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = SCRIPT_DIR / "data" / "features"
IMAGE_OOF_STACK_DIR = FEATURES_DIR / "image_oof_stack"
PREDICTIONS_DIR = IMAGE_OOF_STACK_DIR / "predictions"
MODELS_DIR = IMAGE_OOF_STACK_DIR / "models"

IMAGE_OOF_STACK_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

image_feature_matrix_train = np.load(FEATURES_DIR / "X_image_train.npy")
image_feature_matrix_val = np.load(FEATURES_DIR / "X_image_val.npy")
image_feature_matrix_test = np.load(FEATURES_DIR / "X_image_test.npy")

train_rows_df = pd.read_json(FEATURES_DIR / "train_image_rows.json")
val_rows_df = pd.read_json(FEATURES_DIR / "val_image_rows.json")
test_rows_df = pd.read_json(FEATURES_DIR / "test_image_rows.json")

ratings_train = train_rows_df["rating"].astype(float).values
ratings_val = val_rows_df["rating"].astype(float).values
ratings_test = test_rows_df["rating"].astype(float).values


def clamp_rating_to_scale(predictions):
    return np.clip(predictions, 0.5, 5.0)


def compute_regression_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def write_predictions_csv(path, split_df, y_true, pred, pred_col="y_image_pred"):
    out = pd.DataFrame({
        "id": split_df["id"].values,
        "movie_title": split_df["movie_title"].values,
        "y_true": y_true,
        pred_col: pred,
    })
    out.to_csv(path, index=False)

base_image_models = {
    "dummy_mean": DummyRegressor(strategy="mean"),

    "ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0)
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "extra_trees": ExtraTreesRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "hist_gradient_boosting": HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.025,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_STATE
    ),

    "mlp": make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=5e-4,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=RANDOM_STATE
        )
    ),
}

cv_splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_train_predictions_by_model = {}
val_predictions_by_model = {}
test_predictions_by_model = {}
train_full_predictions_by_model = {}
metric_rows = []

for model_name, model in base_image_models.items():
    print(f"\n==============================")
    print(f"Base model: {model_name}")
    print(f"==============================")

    oof_train_predictions = np.zeros(len(image_feature_matrix_train))
    val_fold_predictions = []
    test_fold_predictions = []

    for fold, (train_index, holdout_index) in enumerate(cv_splitter.split(image_feature_matrix_train), start=1):
        print(f"Training fold {fold}/{N_FOLDS} for {model_name}...")

        fold_fit_model = clone(model)
        fold_fit_model.fit(image_feature_matrix_train[train_index], ratings_train[train_index])

        oof_train_predictions[holdout_index] = clamp_rating_to_scale(
            fold_fit_model.predict(image_feature_matrix_train[holdout_index])
        )
        val_fold_predictions.append(clamp_rating_to_scale(fold_fit_model.predict(image_feature_matrix_val)))
        test_fold_predictions.append(clamp_rating_to_scale(fold_fit_model.predict(image_feature_matrix_test)))

    val_predictions_cv_mean = np.mean(np.column_stack(val_fold_predictions), axis=1)
    test_predictions_cv_mean = np.mean(np.column_stack(test_fold_predictions), axis=1)
    print(f"Training final full-data {model_name}...")
    final_fit_model = clone(model)
    final_fit_model.fit(image_feature_matrix_train, ratings_train)

    train_predictions_full = clamp_rating_to_scale(final_fit_model.predict(image_feature_matrix_train))
    val_predictions_full = clamp_rating_to_scale(final_fit_model.predict(image_feature_matrix_val))
    test_predictions_full = clamp_rating_to_scale(final_fit_model.predict(image_feature_matrix_test))
    oof_train_predictions_by_model[model_name] = oof_train_predictions
    val_predictions_by_model[model_name] = val_predictions_cv_mean
    test_predictions_by_model[model_name] = test_predictions_cv_mean
    train_full_predictions_by_model[model_name] = train_predictions_full

    joblib.dump(final_fit_model, MODELS_DIR / f"{model_name}_final.joblib")

    oof_metrics = compute_regression_metrics(ratings_train, oof_train_predictions)
    val_metrics = compute_regression_metrics(ratings_val, val_predictions_cv_mean)
    test_metrics = compute_regression_metrics(ratings_test, test_predictions_cv_mean)

    full_val_metrics = compute_regression_metrics(ratings_val, val_predictions_full)
    full_test_metrics = compute_regression_metrics(ratings_test, test_predictions_full)

    metric_rows.append({
        "model": model_name,
        "oof_train_mse": oof_metrics["mse"],
        "oof_train_mae": oof_metrics["mae"],
        "oof_train_r2": oof_metrics["r2"],
        "cv_val_mse": val_metrics["mse"],
        "cv_val_mae": val_metrics["mae"],
        "cv_val_r2": val_metrics["r2"],
        "cv_test_mse": test_metrics["mse"],
        "cv_test_mae": test_metrics["mae"],
        "cv_test_r2": test_metrics["r2"],
        "full_val_mse": full_val_metrics["mse"],
        "full_val_mae": full_val_metrics["mae"],
        "full_val_r2": full_val_metrics["r2"],
        "full_test_mse": full_test_metrics["mse"],
        "full_test_mae": full_test_metrics["mae"],
        "full_test_r2": full_test_metrics["r2"],
    })

    write_predictions_csv(
        PREDICTIONS_DIR / f"{model_name}_oof_train_predictions.csv",
        train_rows_df,
        ratings_train,
        oof_train_predictions,
    )
    write_predictions_csv(
        PREDICTIONS_DIR / f"{model_name}_val_predictions.csv",
        val_rows_df,
        ratings_val,
        val_predictions_cv_mean,
    )
    write_predictions_csv(
        PREDICTIONS_DIR / f"{model_name}_test_predictions.csv",
        test_rows_df,
        ratings_test,
        test_predictions_cv_mean,
    )

    print(f"{model_name} OOF train:", oof_metrics)
    print(f"{model_name} CV val:", val_metrics)
    print(f"{model_name} CV test:", test_metrics)

metrics_table = pd.DataFrame(metric_rows).sort_values("cv_val_mse")
metrics_path = IMAGE_OOF_STACK_DIR / "image_base_model_results.csv"
metrics_table.to_csv(metrics_path, index=False)

print("\nBase image model results sorted by CV validation MSE:")

print(metrics_table[[
    "model",
    "oof_train_mse",
    "cv_val_mse",
    "cv_val_mae",
    "cv_val_r2",
    "cv_test_mse",
    "cv_test_mae",
    "cv_test_r2",
]])
print("Saved:", metrics_path)

stacking_model_names = [name for name in base_image_models.keys() if name != "dummy_mean"]

stack_train_matrix = np.column_stack([oof_train_predictions_by_model[name] for name in stacking_model_names])
stack_val_matrix = np.column_stack([val_predictions_by_model[name] for name in stacking_model_names])
stack_test_matrix = np.column_stack([test_predictions_by_model[name] for name in stacking_model_names])

np.save(IMAGE_OOF_STACK_DIR / "Z_image_oof_train.npy", stack_train_matrix)
np.save(IMAGE_OOF_STACK_DIR / "Z_image_val.npy", stack_val_matrix)
np.save(IMAGE_OOF_STACK_DIR / "Z_image_test.npy", stack_test_matrix)

with open(IMAGE_OOF_STACK_DIR / "stack_model_names.json", "w") as f:
    json.dump(stacking_model_names, f, indent=2)

print("\n==============================")
print("Training image OOF stacked ensemble")
print("==============================")

stacking_model = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
)

stacking_model.fit(stack_train_matrix, ratings_train)

stack_train_predictions = clamp_rating_to_scale(stacking_model.predict(stack_train_matrix))
stack_val_predictions = clamp_rating_to_scale(stacking_model.predict(stack_val_matrix))
stack_test_predictions = clamp_rating_to_scale(stacking_model.predict(stack_test_matrix))

stack_train_metrics = compute_regression_metrics(ratings_train, stack_train_predictions)
stack_val_metrics = compute_regression_metrics(ratings_val, stack_val_predictions)
stack_test_metrics = compute_regression_metrics(ratings_test, stack_test_predictions)

print("\nImage OOF Stacked Ensemble")
print("OOF Train:", stack_train_metrics)
print("Val:", stack_val_metrics)
print("Test:", stack_test_metrics)

write_predictions_csv(
    PREDICTIONS_DIR / "image_oof_stacked_ensemble_train_predictions.csv",
    train_rows_df,
    ratings_train,
    stack_train_predictions,
)
write_predictions_csv(
    PREDICTIONS_DIR / "image_oof_stacked_ensemble_val_predictions.csv",
    val_rows_df,
    ratings_val,
    stack_val_predictions,
)
write_predictions_csv(
    PREDICTIONS_DIR / "image_oof_stacked_ensemble_test_predictions.csv",
    test_rows_df,
    ratings_test,
    stack_test_predictions,
)

joblib.dump(stacking_model, MODELS_DIR / "image_oof_stacker.joblib")

stacker_ridgecv = stacking_model.named_steps["ridgecv"]
stacking_weights_table = pd.DataFrame({
    "base_image_model": stacking_model_names,
    "stacking_weight": stacker_ridgecv.coef_,
})
stacking_weights_table["intercept"] = stacker_ridgecv.intercept_
stacking_weights_table.to_csv(IMAGE_OOF_STACK_DIR / "image_oof_stacking_weights.csv", index=False)

print("\nStacking weights:")
print(stacking_weights_table)

stack_ensemble_summary = {
    "model": "image_oof_stacked_ensemble",
    "oof_train_mse": stack_train_metrics["mse"],
    "oof_train_mae": stack_train_metrics["mae"],
    "oof_train_r2": stack_train_metrics["r2"],
    "val_mse": stack_val_metrics["mse"],
    "val_mae": stack_val_metrics["mae"],
    "val_r2": stack_val_metrics["r2"],
    "test_mse": stack_test_metrics["mse"],
    "test_mae": stack_test_metrics["mae"],
    "test_r2": stack_test_metrics["r2"],
}
summary_path = IMAGE_OOF_STACK_DIR / "image_oof_stacked_ensemble_results.json"
with open(summary_path, "w") as f:
    json.dump(stack_ensemble_summary, f, indent=2)

print("\nSaved OOF stacked ensemble results:", summary_path)
print("Saved OOF stacked ensemble predictions in:", PREDICTIONS_DIR)