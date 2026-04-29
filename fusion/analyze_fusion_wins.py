import pandas as pd
import numpy as np

def find_visual_disambiguation_wins(predictions_path, top_n=10):
    print(f"Loading predictions from {predictions_path}...\n")
    df = pd.read_csv(predictions_path)
    
    # Calculate the absolute error for the Text-Only model
    df['text_error'] = np.abs(df['y_true'] - df['y_text_pred'])
    
    # Calculate the absolute error for the winning Multi-modal model (HGB)
    df['fusion_error'] = np.abs(df['y_true'] - df['y_hgb_fusion_pred'])
    
    # Calculate how much the image "helped" (Positive number = Fusion was better)
    df['image_help'] = df['text_error'] - df['fusion_error']
    
    # Filter for rows where the Text model made a significant mistake (e.g., off by more than 0.5 stars)
    # AND where the Fusion model successfully reduced that error
    meaningful_mistakes = df[(df['text_error'] > 0.5) & (df['image_help'] > 0)]
    
    # Sort by the most helpful image interventions
    top_wins = meaningful_mistakes.sort_values(by='image_help', ascending=False).head(top_n)
    
    print(f"--- Top {top_n} Visual Disambiguation Wins ---")
    print("These are the test samples where the CLIP+ResNet image features most successfully")
    print("corrected the Text model's predictions.\n")
    
    for i, row in enumerate(top_wins.itertuples(), 1):
        print(f"#{i} - {row.movie_title}")
        print(f"  True Rating:      {row.y_true:.2f}")
        print(f"  Text Pred:        {row.y_text_pred:.2f} (Error: {row.text_error:.2f})")
        print(f"  Image Pred:       {row.y_image_pred:.2f}")
        print(f"  HGB Fusion Pred:  {row.y_hgb_fusion_pred:.2f} (Error: {row.fusion_error:.2f})")
        print(f"  Visual Net Help: +{row.image_help:.2f} stars improvement")
        print("-" * 50)
        
    # Optional: Save these top wins to a CSV for easy copy-pasting into your report
    top_wins.to_csv('fusion/results/qualitative_audit_wins.csv', index=False)
    print("\nSaved full details to: fusion/results/qualitative_audit_wins.csv")

if __name__ == "__main__":
    # Update this path if your file is located somewhere else
    PREDICTIONS_FILE = "fusion/predictions/fusion_test_predictions.csv"
    find_visual_disambiguation_wins(PREDICTIONS_FILE, top_n=10)