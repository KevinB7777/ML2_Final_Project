import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs("data/features/figures", exist_ok=True)

df = pd.read_json("data/features/train_image_rows.json")

sample = df.sample(n=16, random_state=42)

fig, axes = plt.subplots(4, 4, figsize=(10, 12))

for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
    img = Image.open(row["poster_path"]).convert("RGB")
    ax.imshow(img)
    ax.axis("off")
    title = row["movie_title"]
    rating = row["rating"]
    ax.set_title(f"{title[:22]}\nRating: {rating}", fontsize=8)

plt.tight_layout()
out_path = "data/features/figures/poster_sample_grid.png"
plt.savefig(out_path, dpi=250)
print("Saved:", out_path)
plt.show()