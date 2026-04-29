import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = SCRIPT_DIR / "data" / "features"
FIGURES_DIR = FEATURES_DIR / "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

train_feature_matrix = np.load(FEATURES_DIR / "X_image_train.npy")
train_rows_df = pd.read_json(FEATURES_DIR / "train_image_rows.json")

ratings_train = train_rows_df["rating"].astype(float).values
num_rows_to_plot = min(2000, len(train_rows_df))
feature_matrix_sample = train_feature_matrix[:num_rows_to_plot]
ratings_sample = ratings_train[:num_rows_to_plot]

pca_model = PCA(n_components=2, random_state=42)
pca_projection_2d = pca_model.fit_transform(feature_matrix_sample)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    pca_projection_2d[:, 0],
    pca_projection_2d[:, 1],
    c=ratings_sample,
    alpha=0.65,
    s=18
)
plt.colorbar(scatter, label="Rating")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("ResNet18 Poster Embeddings Projected with PCA")
plt.tight_layout()

pca_plot_path = FIGURES_DIR / "poster_embedding_pca.png"
plt.savefig(pca_plot_path, dpi=250)
print("Saved:", pca_plot_path)
plt.show()