import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
from pathlib import Path


def prepare_dataset(raw_json_path, output_dir, sample_size=8000):
    print(f"Loading raw data from {raw_json_path} ...")
    df = pd.read_json(raw_json_path)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['review_detail', 'rating', 'movie'])
    df['rating'] = df['rating'].astype(float) / 2.0
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    contract_df = pd.DataFrame({
        "id":          df['review_id'].astype(str),
        "review_text": df['review_detail'].astype(str),
        "image_path":  df['movie'].astype(str),
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


def prepare_all(input_dir, output_dir, sample_size=8000):
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
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['review_detail', 'rating', 'movie'])
    print(f"Row count after dropping nulls: {len(df)}")
    before = len(df)
    df = df.drop_duplicates(subset=['review_id'])
    if len(df) < before:
        print(f"Removed {before - len(df)} duplicate review_id rows.")
    df['rating'] = df['rating'].astype(float) / 2.0

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    contract_df = pd.DataFrame({
        "id":          df['review_id'].astype(str),
        "review_text": df['review_detail'].astype(str),
        "image_path":  df['movie'].astype(str),
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
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    prepare_all(DATA_DIR / 'raw', DATA_DIR / 'processed')