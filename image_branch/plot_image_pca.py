import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

os.makedirs("data/features/figures", exist_ok=True)

X = np.load("data/features/X_image_train.npy")
df = pd.read_json("data/features/train_image_rows.json")

ratings = df["rating"].astype(float).values

# Keep plot readable
n = min(2000, len(df))
X_sample = X[:n]
ratings_sample = ratings[:n]

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_sample)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=ratings_sample,
    alpha=0.65,
    s=18
)
plt.colorbar(scatter, label="Rating")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("ResNet18 Poster Embeddings Projected with PCA")
plt.tight_layout()

out_path = "data/features/figures/poster_embedding_pca.png"
plt.savefig(out_path, dpi=250)
print("Saved:", out_path)
plt.show()