import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob


def prepare_dataset(raw_json_path, output_dir, sample_size=8000):
    """
    Cleans raw IMDb data (JSON format) and enforces the shared JSON contract.

    Expected JSON schema per record:
        review_id     : str
        reviewer      : str
        movie         : str
        rating        : str | null   (IMDb 1-10 scale; null means unrated)
        review_summary: str
        review_date   : str
        spoiler_tag   : int
        review_detail : str
        helpful       : [str, str]   (helpful_votes, total_votes)
    """
    print(f"Loading raw data from {raw_json_path} ...")
    df = pd.read_json(raw_json_path)   # <-- JSON array, not CSV

    # 'rating' arrives as a string (e.g. "8") or None/NaN.
    # Coerce to numeric first so dropna works correctly and bad values
    # (non-numeric strings, if any) become NaN and get dropped cleanly.
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop rows missing crucial data
    df = df.dropna(subset=['review_detail', 'rating', 'movie'])

    # Normalize IMDb 1-10 rating to 0.5-5.0 scale (÷2) to match Letterboxd
    df['rating'] = df['rating'].astype(float) / 2.0

    # Sample down to `sample_size` to keep feature extraction fast
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Enforce the agreed JSON contract structure
    contract_df = pd.DataFrame({
        "id":          df['review_id'].astype(str),          # stable unique id from source
        "review_text": df['review_detail'].astype(str),
        "image_path":  df['movie'].astype(str),              # partner queries TMDb with this
        "rating":      df['rating'].astype(float),
    })

    # 80 / 10 / 10  Train / Val / Test split
    train_df, temp_df = train_test_split(contract_df, test_size=0.2, random_state=42)
    val_df,  test_df  = train_test_split(temp_df,     test_size=0.5, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_json(os.path.join(output_dir, 'train.json'), orient='records', indent=2)
    val_df  .to_json(os.path.join(output_dir, 'val.json'),   orient='records', indent=2)
    test_df .to_json(os.path.join(output_dir, 'test.json'),  orient='records', indent=2)

    print(
        f"Saved to '{output_dir}'. "
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def prepare_all(input_dir, output_dir, sample_size=8000):
    """
    Loads all JSON files in `input_dir`, concatenates them, then runs the
    same cleaning / splitting pipeline as `prepare_dataset`.

    Use this when your 8 raw files should be treated as one combined dataset.
    """
    json_files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir!r}")

    print(f"Found {len(json_files)} file(s): {[os.path.basename(f) for f in json_files]}")

    frames = []
    for path in json_files:
        print(f"  Reading {os.path.basename(path)} ...")
        frames.append(pd.read_json(path))

    df = pd.concat(frames, ignore_index=True)
    print(f"Combined row count before cleaning: {len(df)}")

    # Coerce rating to numeric (string "8" → 8.0, null → NaN)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop rows missing crucial data
    df = df.dropna(subset=['review_detail', 'rating', 'movie'])
    print(f"Row count after dropping nulls: {len(df)}")

    # Deduplicate on review_id in case files overlap
    before = len(df)
    df = df.drop_duplicates(subset=['review_id'])
    if len(df) < before:
        print(f"Removed {before - len(df)} duplicate review_id rows.")

    # Normalize IMDb 1-10 → 0.5-5.0
    df['rating'] = df['rating'].astype(float) / 2.0

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    contract_df = pd.DataFrame({
        "id":          df['review_id'].astype(str),
        "review_text": df['review_detail'].astype(str),
        "movie_title":  df['movie'].astype(str),
        "rating":      df['rating'].astype(float),
    })

    train_df, temp_df = train_test_split(contract_df, test_size=0.2, random_state=42)
    val_df,  test_df  = train_test_split(temp_df,     test_size=0.5, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_json(os.path.join(output_dir, 'train.json'), orient='records', indent=2)
    val_df  .to_json(os.path.join(output_dir, 'val.json'),   orient='records', indent=2)
    test_df .to_json(os.path.join(output_dir, 'test.json'),  orient='records', indent=2)

    print(
        f"Saved to '{output_dir}'. "
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


if __name__ == "__main__":
    # --- Single file (mirrors your original usage) ---
    # prepare_dataset('data/raw/imdb_reviews.json', 'data/processed/')

    # --- All 8 files at once (recommended) ---
    prepare_all('data/raw/', 'data/processed/')