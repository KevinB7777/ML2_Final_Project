import json
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = SCRIPT_DIR / "data" / "features"
IMAGE_OOF_STACK_DIR = FEATURES_DIR / "image_oof_stack"

base_results_path = IMAGE_OOF_STACK_DIR / "image_base_model_results.csv"
ensemble_summary_path = IMAGE_OOF_STACK_DIR / "image_oof_stacked_ensemble_results.json"

base_results_df = pd.read_csv(base_results_path)

with open(ensemble_summary_path, "r") as f:
    ensemble_summary = json.load(f)

summary_rows = []

for _, base_model_row in base_results_df.iterrows():
    summary_rows.append({
        "Model": base_model_row["model"],
        "Val MSE": base_model_row["cv_val_mse"],
        "Val MAE": base_model_row["cv_val_mae"],
        "Val R2": base_model_row["cv_val_r2"],
        "Test MSE": base_model_row["cv_test_mse"],
        "Test MAE": base_model_row["cv_test_mae"],
        "Test R2": base_model_row["cv_test_r2"],
    })

summary_rows.append({
    "Model": "image_oof_stacked_ensemble",
    "Val MSE": ensemble_summary["val_mse"],
    "Val MAE": ensemble_summary["val_mae"],
    "Val R2": ensemble_summary["val_r2"],
    "Test MSE": ensemble_summary["test_mse"],
    "Test MAE": ensemble_summary["test_mae"],
    "Test R2": ensemble_summary["test_r2"],
})

summary_table = pd.DataFrame(summary_rows).sort_values("Val MSE")
summary_csv_path = FEATURES_DIR / "image_branch_summary_table.csv"
summary_table.to_csv(summary_csv_path, index=False)

print(summary_table.round(4))
print("\nSaved:", summary_csv_path)