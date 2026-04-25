import os
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


# --------------------------------------------------
# Config
# --------------------------------------------------
N_FOLDS = 5
RANDOM_STATE = 42

FEATURE_DIR = "data/features"
OUT_DIR = "data/features/image_oof_stack"
PRED_DIR = os.path.join(OUT_DIR, "predictions")
MODEL_DIR = os.path.join(OUT_DIR, "models")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------
X_train = np.load("data/features/X_image_train.npy")
X_val = np.load("data/features/X_image_val.npy")
X_test = np.load("data/features/X_image_test.npy")

train_df = pd.read_json("data/features/train_image_rows.json")
val_df = pd.read_json("data/features/val_image_rows.json")
test_df = pd.read_json("data/features/test_image_rows.json")

y_train = train_df["rating"].astype(float).values
y_val = val_df["rating"].astype(float).values
y_test = test_df["rating"].astype(float).values


def clip_rating(pred):
    return np.clip(pred, 0.5, 5.0)


def compute_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def save_pred_csv(path, df, y_true, pred, pred_col="y_image_pred"):
    out = pd.DataFrame({
        "id": df["id"].values,
        "movie_title": df["movie_title"].values,
        "y_true": y_true,
        pred_col: pred,
    })
    out.to_csv(path, index=False)


# --------------------------------------------------
# Base image models
# --------------------------------------------------
base_models = {
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


# --------------------------------------------------
# OOF predictions
# --------------------------------------------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = {}
val_preds = {}
test_preds = {}
train_full_preds = {}
model_metrics = []

for model_name, model in base_models.items():
    print(f"\n==============================")
    print(f"Base model: {model_name}")
    print(f"==============================")

    oof = np.zeros(len(X_train))
    fold_val_preds = []
    fold_test_preds = []

    for fold, (tr_idx, hold_idx) in enumerate(kf.split(X_train), start=1):
        print(f"Training fold {fold}/{N_FOLDS} for {model_name}...")

        fold_model = clone(model)
        fold_model.fit(X_train[tr_idx], y_train[tr_idx])

        oof[hold_idx] = clip_rating(fold_model.predict(X_train[hold_idx]))
        fold_val_preds.append(clip_rating(fold_model.predict(X_val)))
        fold_test_preds.append(clip_rating(fold_model.predict(X_test)))

    # Average fold models' predictions for validation/test
    val_pred_cv = np.mean(np.column_stack(fold_val_preds), axis=1)
    test_pred_cv = np.mean(np.column_stack(fold_test_preds), axis=1)

    # Also fit a final full-train model for a standard train/val/test comparison
    print(f"Training final full-data {model_name}...")
    final_model = clone(model)
    final_model.fit(X_train, y_train)

    train_pred_full = clip_rating(final_model.predict(X_train))
    val_pred_full = clip_rating(final_model.predict(X_val))
    test_pred_full = clip_rating(final_model.predict(X_test))

    # Store predictions used for stacking
    oof_preds[model_name] = oof
    val_preds[model_name] = val_pred_cv
    test_preds[model_name] = test_pred_cv
    train_full_preds[model_name] = train_pred_full

    # Save final model
    joblib.dump(final_model, os.path.join(MODEL_DIR, f"{model_name}_final.joblib"))

    # Metrics
    oof_m = compute_metrics(y_train, oof)
    val_m = compute_metrics(y_val, val_pred_cv)
    test_m = compute_metrics(y_test, test_pred_cv)

    full_val_m = compute_metrics(y_val, val_pred_full)
    full_test_m = compute_metrics(y_test, test_pred_full)

    model_metrics.append({
        "model": model_name,
        "oof_train_mse": oof_m["mse"],
        "oof_train_mae": oof_m["mae"],
        "oof_train_r2": oof_m["r2"],
        "cv_val_mse": val_m["mse"],
        "cv_val_mae": val_m["mae"],
        "cv_val_r2": val_m["r2"],
        "cv_test_mse": test_m["mse"],
        "cv_test_mae": test_m["mae"],
        "cv_test_r2": test_m["r2"],
        "full_val_mse": full_val_m["mse"],
        "full_val_mae": full_val_m["mae"],
        "full_val_r2": full_val_m["r2"],
        "full_test_mse": full_test_m["mse"],
        "full_test_mae": full_test_m["mae"],
        "full_test_r2": full_test_m["r2"],
    })

    # Save predictions
    save_pred_csv(
        os.path.join(PRED_DIR, f"{model_name}_oof_train_predictions.csv"),
        train_df,
        y_train,
        oof,
    )
    save_pred_csv(
        os.path.join(PRED_DIR, f"{model_name}_val_predictions.csv"),
        val_df,
        y_val,
        val_pred_cv,
    )
    save_pred_csv(
        os.path.join(PRED_DIR, f"{model_name}_test_predictions.csv"),
        test_df,
        y_test,
        test_pred_cv,
    )

    print(f"{model_name} OOF train:", oof_m)
    print(f"{model_name} CV val:", val_m)
    print(f"{model_name} CV test:", test_m)


# --------------------------------------------------
# Save base-model metrics
# --------------------------------------------------
metrics_df = pd.DataFrame(model_metrics).sort_values("cv_val_mse")
metrics_path = os.path.join(OUT_DIR, "image_base_model_results.csv")
metrics_df.to_csv(metrics_path, index=False)

print("\nBase image model results sorted by CV validation MSE:")
print(metrics_df[[
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


# --------------------------------------------------
# Build stacking matrices
# --------------------------------------------------
stack_model_names = [name for name in base_models.keys() if name != "dummy_mean"]

Z_train = np.column_stack([oof_preds[name] for name in stack_model_names])
Z_val = np.column_stack([val_preds[name] for name in stack_model_names])
Z_test = np.column_stack([test_preds[name] for name in stack_model_names])

np.save(os.path.join(OUT_DIR, "Z_image_oof_train.npy"), Z_train)
np.save(os.path.join(OUT_DIR, "Z_image_val.npy"), Z_val)
np.save(os.path.join(OUT_DIR, "Z_image_test.npy"), Z_test)

with open(os.path.join(OUT_DIR, "stack_model_names.json"), "w") as f:
    json.dump(stack_model_names, f, indent=2)


# --------------------------------------------------
# Image stacked ensemble
# --------------------------------------------------
print("\n==============================")
print("Training image OOF stacked ensemble")
print("==============================")

stacker = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
)

stacker.fit(Z_train, y_train)

stack_train_pred = clip_rating(stacker.predict(Z_train))
stack_val_pred = clip_rating(stacker.predict(Z_val))
stack_test_pred = clip_rating(stacker.predict(Z_test))

stack_train_m = compute_metrics(y_train, stack_train_pred)
stack_val_m = compute_metrics(y_val, stack_val_pred)
stack_test_m = compute_metrics(y_test, stack_test_pred)

print("\nImage OOF Stacked Ensemble")
print("OOF Train:", stack_train_m)
print("Val:", stack_val_m)
print("Test:", stack_test_m)

save_pred_csv(
    os.path.join(PRED_DIR, "image_oof_stacked_ensemble_train_predictions.csv"),
    train_df,
    y_train,
    stack_train_pred,
)
save_pred_csv(
    os.path.join(PRED_DIR, "image_oof_stacked_ensemble_val_predictions.csv"),
    val_df,
    y_val,
    stack_val_pred,
)
save_pred_csv(
    os.path.join(PRED_DIR, "image_oof_stacked_ensemble_test_predictions.csv"),
    test_df,
    y_test,
    stack_test_pred,
)

joblib.dump(stacker, os.path.join(MODEL_DIR, "image_oof_stacker.joblib"))

# Save stacker weights
ridgecv = stacker.named_steps["ridgecv"]
weights_df = pd.DataFrame({
    "base_image_model": stack_model_names,
    "stacking_weight": ridgecv.coef_,
})
weights_df["intercept"] = ridgecv.intercept_
weights_df.to_csv(os.path.join(OUT_DIR, "image_oof_stacking_weights.csv"), index=False)

print("\nStacking weights:")
print(weights_df)

# Save summary including ensemble
ensemble_summary = {
    "model": "image_oof_stacked_ensemble",
    "oof_train_mse": stack_train_m["mse"],
    "oof_train_mae": stack_train_m["mae"],
    "oof_train_r2": stack_train_m["r2"],
    "val_mse": stack_val_m["mse"],
    "val_mae": stack_val_m["mae"],
    "val_r2": stack_val_m["r2"],
    "test_mse": stack_test_m["mse"],
    "test_mae": stack_test_m["mae"],
    "test_r2": stack_test_m["r2"],
}

summary_path = os.path.join(OUT_DIR, "image_oof_stacked_ensemble_results.json")
with open(summary_path, "w") as f:
    json.dump(ensemble_summary, f, indent=2)

print("\nSaved OOF stacked ensemble results:", summary_path)
print("Saved OOF stacked ensemble predictions in:", PRED_DIR)