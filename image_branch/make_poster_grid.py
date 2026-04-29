import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = SCRIPT_DIR / "data" / "features"
FIGURES_DIR = FEATURES_DIR / "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

train_rows_df = pd.read_json(FEATURES_DIR / "train_image_rows.json")

sample_rows_df = train_rows_df.sample(n=16, random_state=42)

fig, axes = plt.subplots(4, 4, figsize=(10, 12))

for ax, (_, sample_row) in zip(axes.flatten(), sample_rows_df.iterrows()):
    poster_path = Path(sample_row["poster_path"])
    if not poster_path.is_absolute():
        poster_path = SCRIPT_DIR / poster_path

    img = Image.open(poster_path).convert("RGB")
    ax.imshow(img)
    ax.axis("off")
    title = sample_row["movie_title"]
    rating = sample_row["rating"]
    ax.set_title(f"{title[:22]}\nRating: {rating}", fontsize=8)

plt.tight_layout()
poster_grid_path = FIGURES_DIR / "poster_sample_grid.png"
plt.savefig(poster_grid_path, dpi=250)
print("Saved:", poster_grid_path)
plt.show()