import pandas as pd
import numpy as np
from pathlib import Path


def find_all_visual_help_cases(predictions_path, top_n=10):
    predictions_path = Path(predictions_path)

    print(f"Loading predictions from {predictions_path}...\n")
    df = pd.read_csv(predictions_path)

    required_cols = {
        "id",
        "movie_title",
        "y_true",
        "y_text_pred",
        "y_image_pred",
        "y_hgb_fusion_pred",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["text_error"] = np.abs(df["y_true"] - df["y_text_pred"])
    df["fusion_error"] = np.abs(df["y_true"] - df["y_hgb_fusion_pred"])

    df["image_help"] = df["text_error"] - df["fusion_error"]

    helped = df[df["image_help"] > 0].copy()
    helped = helped.sort_values(by="image_help", ascending=False)

    meaningful_helped = helped[helped["text_error"] > 0.5].copy()

    out_dir = Path("fusion/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_help_path = out_dir / "qualitative_audit_all_visual_help_cases.csv"
    meaningful_help_path = out_dir / "qualitative_audit_meaningful_visual_help_cases.csv"
    top_help_path = out_dir / "qualitative_audit_top_visual_help_cases.csv"

    helped.to_csv(all_help_path, index=False)
    meaningful_helped.to_csv(meaningful_help_path, index=False)
    helped.head(top_n).to_csv(top_help_path, index=False)

    print("--- Visual Help Summary ---")
    print(f"Total test rows: {len(df)}")
    print(f"Rows where fusion improved over text-only: {len(helped)}")
    print(f"Rows where text_error > 0.5 and fusion improved: {len(meaningful_helped)}")
    print(f"Average text error: {df['text_error'].mean():.4f}")
    print(f"Average fusion error: {df['fusion_error'].mean():.4f}")
    print(f"Average improvement across all rows: {df['image_help'].mean():.4f}")
    print(f"Average improvement among helped rows: {helped['image_help'].mean():.4f}")

    print(f"\n--- Top {top_n} Visual Help Cases ---")
    for i, row in enumerate(helped.head(top_n).itertuples(), 1):
        print(f"#{i} - {row.movie_title}")
        print(f"  True Rating:      {row.y_true:.2f}")
        print(f"  Text Pred:        {row.y_text_pred:.2f} (Error: {row.text_error:.2f})")
        print(f"  Image Pred:       {row.y_image_pred:.2f}")
        print(f"  HGB Fusion Pred:  {row.y_hgb_fusion_pred:.2f} (Error: {row.fusion_error:.2f})")
        print(f"  Visual Help:     +{row.image_help:.2f} stars improvement")
        print("-" * 50)

    print("\nSaved:")
    print(all_help_path)
    print(meaningful_help_path)
    print(top_help_path)


if __name__ == "__main__":
    predictions_file = "fusion/predictions/fusion_test_predictions.csv"
    find_all_visual_help_cases(predictions_file, top_n=10)