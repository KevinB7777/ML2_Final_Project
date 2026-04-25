import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from text_features import process_text_features
import joblib
import os

def train_text_model():
    print("Extracting features...")
    X_train, y_train = process_text_features('data/processed/train.json', is_train=True)
    X_val, y_val = process_text_features('data/processed/val.json', is_train=False)
    
    print("Training Ridge Regressor...")
    # Built-in cross-validation for L2 penalty tuning (handles bias-variance tradeoff)
    alphas = np.logspace(-3, 3, 10)
    model = RidgeCV(alphas=alphas)
    model.fit(X_train, y_train)
    
    print("Evaluating on Validation Set...")
    y_pred = model.predict(X_val)
    
    # Contract Enforcement
    assert y_pred.shape == y_val.shape, "Contract Failed: y_pred shape mismatch."
    assert len(y_pred.shape) == 1, "Contract Failed: y_pred must be 1D."
    
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    
    print("\n--- Text Expert Baseline Results ---")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Save the model and the predictions for the late fusion stage
    os.makedirs('fusion', exist_ok=True)
    joblib.dump(model, 'text/ridge_model.pkl')
    np.save('fusion/y_text_pred_val.npy', y_pred)
    print("Saved predictions to fusion/y_text_pred_val.npy")

if __name__ == "__main__":
    train_text_model()