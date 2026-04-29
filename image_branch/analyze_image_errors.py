from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = SCRIPT_DIR / "data" / "features"
IMAGE_OOF_STACK_DIR = FEATURES_DIR / "image_oof_stack"
PREDICTIONS_DIR = IMAGE_OOF_STACK_DIR / "predictions"
FIGURES_DIR = FEATURES_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

test_pred_path = PREDICTIONS_DIR / "image_oof_stacked_ensemble_test_predictions.csv"
test_predictions_df = pd.read_csv(test_pred_path)

test_predictions_df["abs_error"] = (test_predictions_df["y_true"] - test_predictions_df["y_image_pred"]).abs()
test_predictions_df["signed_error"] = test_predictions_df["y_image_pred"] - test_predictions_df["y_true"]

print("\nBest predictions:")
print(test_predictions_df.sort_values("abs_error").head(10)[["movie_title", "y_true", "y_image_pred", "abs_error"]])

print("\nWorst predictions:")
print(test_predictions_df.sort_values("abs_error", ascending=False).head(10)[["movie_title", "y_true", "y_image_pred", "abs_error"]])

test_predictions_df.sort_values("abs_error").head(20).to_csv(
    IMAGE_OOF_STACK_DIR / "best_image_predictions.csv",
    index=False
)

test_predictions_df.sort_values("abs_error", ascending=False).head(20).to_csv(
    IMAGE_OOF_STACK_DIR / "worst_image_predictions.csv",
    index=False
)

plt.figure(figsize=(8, 6))
plt.scatter(test_predictions_df["y_true"], test_predictions_df["y_image_pred"], alpha=0.65)
plt.xlabel("True Rating")
plt.ylabel("Predicted Rating")
plt.title("Image-Only Stacked Ensemble: True vs Predicted Ratings")
plt.tight_layout()
scatter_plot_path = FIGURES_DIR / "image_true_vs_pred.png"
plt.savefig(scatter_plot_path, dpi=250)
print("Saved:", scatter_plot_path)

plt.figure(figsize=(8, 6))
plt.hist(test_predictions_df["abs_error"], bins=30)
plt.xlabel("Absolute Error")
plt.ylabel("Count")
plt.title("Image-Only Stacked Ensemble Absolute Error Distribution")
plt.tight_layout()
abs_error_hist_path = FIGURES_DIR / "image_abs_error_hist.png"
plt.savefig(abs_error_hist_path, dpi=250)
print("Saved:", abs_error_hist_path)

plt.show()