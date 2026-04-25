import os
import re
import time
import json
import requests
import pandas as pd
from tqdm import tqdm

API_KEY = os.environ.get("TVDB_API_KEY")

if not API_KEY:
    raise ValueError("Missing TVDB_API_KEY. Run: export TVDB_API_KEY='your_key'")

BASE_URL = "https://api4.thetvdb.com/v4"

POSTER_DIR = "data/posters"
CACHE_PATH = "data/tvdb_poster_cache.json"

os.makedirs(POSTER_DIR, exist_ok=True)


def login():
    url = f"{BASE_URL}/login"
    payload = {"apikey": API_KEY}

    response = requests.post(url, json=payload, timeout=15)
    response.raise_for_status()

    data = response.json()
    return data["data"]["token"]


def clean_query_title(movie_title):
    """
    Examples:
    'The Matrix Reloaded (2003)' -> 'The Matrix Reloaded'
    'Mission: Impossible (1966–1973)' -> 'Mission: Impossible'
    'Sahara (I) (1943)' -> 'Sahara'
    """
    title = str(movie_title)

    # Remove all parenthetical parts
    title = re.sub(r"\([^)]*\)", "", title)

    # Normalize spaces
    title = re.sub(r"\s+", " ", title).strip()

    return title


def extract_year(movie_title):
    """
    Extracts first 4-digit year from title if available.
    'The Matrix Reloaded (2003)' -> '2003'
    """
    match = re.search(r"(18|19|20)\d{2}", str(movie_title))
    if match:
        return match.group(0)
    return None


def safe_filename(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:120]


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def search_tvdb(token, movie_title):
    """
    Search TheTVDB by movie title.
    Returns best image_url or None.
    """
    query = clean_query_title(movie_title)
    expected_year = extract_year(movie_title)

    url = f"{BASE_URL}/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "query": query,
        "type": "movie"
    }

    response = requests.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()

    data = response.json().get("data", [])

    if not data:
        return None

    # Prefer exact year match if available
    if expected_year:
        for item in data:
            if str(item.get("year")) == expected_year and item.get("image_url"):
                return item.get("image_url")

    # Otherwise use first result with image_url
    for item in data:
        if item.get("image_url"):
            return item.get("image_url")

    return None


def download_image(url, output_path):
    if os.path.exists(output_path):
        return True

    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        if "image" not in content_type.lower():
            print("Not image:", url, content_type)
            return False

        with open(output_path, "wb") as f:
            f.write(response.content)

        return True

    except Exception as e:
        print("Image download failed:", url, e)
        return False


def get_poster_url_for_title(token, movie_title, cache):
    if movie_title in cache:
        return cache[movie_title]

    try:
        poster_url = search_tvdb(token, movie_title)
        cache[movie_title] = poster_url
        save_cache(cache)

        # Small delay to be polite
        time.sleep(0.05)

        return poster_url

    except Exception as e:
        print("TVDB search failed:", movie_title, e)
        cache[movie_title] = None
        save_cache(cache)
        return None


def process_split(split, token, cache):
    input_path = f"data/processed/{split}.json"
    df = pd.read_json(input_path)

    if "movie_title" not in df.columns and "image_path" in df.columns:
        df = df.rename(columns={"image_path": "movie_title"})

    poster_urls = []
    poster_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
        movie_title = row["movie_title"]

        poster_url = get_poster_url_for_title(token, movie_title, cache)
        poster_urls.append(poster_url)

        if poster_url is None:
            poster_paths.append(None)
            continue

        filename = safe_filename(movie_title + "_" + str(row["id"])) + ".jpg"
        output_path = os.path.join(POSTER_DIR, filename)

        ok = download_image(poster_url, output_path)

        if ok:
            poster_paths.append(output_path)
        else:
            poster_paths.append(None)

    df["poster_url"] = poster_urls
    df["poster_path"] = poster_paths

    before = len(df)
    df = df.dropna(subset=["poster_path"])
    after = len(df)

    output_path = f"data/processed/{split}_with_posters_tvdb.json"
    df.to_json(output_path, orient="records", indent=2)

    print(f"{split}: saved {after}/{before} rows with posters")
    print("Saved:", output_path)


def main():
    token = login()
    print("Logged into TheTVDB successfully.")

    cache = load_cache()

    for split in ["train", "val", "test"]:
        process_split(split, token, cache)

    print("Done.")


if __name__ == "__main__":
    main()