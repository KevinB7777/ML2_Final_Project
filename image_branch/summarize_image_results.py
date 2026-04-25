import json
import pandas as pd

base_path = "data/features/image_oof_stack/image_base_model_results.csv"
ensemble_path = "data/features/image_oof_stack/image_oof_stacked_ensemble_results.json"

base = pd.read_csv(base_path)

with open(ensemble_path, "r") as f:
    ensemble = json.load(f)

rows = []

for _, row in base.iterrows():
    rows.append({
        "Model": row["model"],
        "Val MSE": row["cv_val_mse"],
        "Val MAE": row["cv_val_mae"],
        "Val R2": row["cv_val_r2"],
        "Test MSE": row["cv_test_mse"],
        "Test MAE": row["cv_test_mae"],
        "Test R2": row["cv_test_r2"],
    })

rows.append({
    "Model": "image_oof_stacked_ensemble",
    "Val MSE": ensemble["val_mse"],
    "Val MAE": ensemble["val_mae"],
    "Val R2": ensemble["val_r2"],
    "Test MSE": ensemble["test_mse"],
    "Test MAE": ensemble["test_mae"],
    "Test R2": ensemble["test_r2"],
})

summary = pd.DataFrame(rows).sort_values("Val MSE")
summary.to_csv("data/features/image_branch_summary_table.csv", index=False)

print(summary.round(4))
print("\nSaved: data/features/image_branch_summary_table.csv")