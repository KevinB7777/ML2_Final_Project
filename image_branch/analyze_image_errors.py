import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("data/features/figures", exist_ok=True)

pred_path = "data/features/image_oof_stack/predictions/image_oof_stacked_ensemble_test_predictions.csv"
df = pd.read_csv(pred_path)

df["abs_error"] = (df["y_true"] - df["y_image_pred"]).abs()
df["signed_error"] = df["y_image_pred"] - df["y_true"]

print("\nBest predictions:")
print(df.sort_values("abs_error").head(10)[["movie_title", "y_true", "y_image_pred", "abs_error"]])

print("\nWorst predictions:")
print(df.sort_values("abs_error", ascending=False).head(10)[["movie_title", "y_true", "y_image_pred", "abs_error"]])

df.sort_values("abs_error").head(20).to_csv(
    "data/features/image_oof_stack/best_image_predictions.csv",
    index=False
)

df.sort_values("abs_error", ascending=False).head(20).to_csv(
    "data/features/image_oof_stack/worst_image_predictions.csv",
    index=False
)

plt.figure(figsize=(8, 6))
plt.scatter(df["y_true"], df["y_image_pred"], alpha=0.65)
plt.xlabel("True Rating")
plt.ylabel("Predicted Rating")
plt.title("Image-Only Stacked Ensemble: True vs Predicted Ratings")
plt.tight_layout()
plt.savefig("data/features/figures/image_true_vs_pred.png", dpi=250)
print("Saved: data/features/figures/image_true_vs_pred.png")

plt.figure(figsize=(8, 6))
plt.hist(df["abs_error"], bins=30)
plt.xlabel("Absolute Error")
plt.ylabel("Count")
plt.title("Image-Only Stacked Ensemble Absolute Error Distribution")
plt.tight_layout()
plt.savefig("data/features/figures/image_abs_error_hist.png", dpi=250)
print("Saved: data/features/figures/image_abs_error_hist.png")

plt.show()