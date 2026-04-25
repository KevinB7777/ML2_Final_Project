import os
import requests
import json

API_KEY = os.environ.get("TVDB_API_KEY")

if not API_KEY:
    raise ValueError("Missing TVDB_API_KEY. Run: export TVDB_API_KEY='your_key'")

BASE_URL = "https://api4.thetvdb.com/v4"


def login():
    url = f"{BASE_URL}/login"

    payload = {
        "apikey": API_KEY
    }

    r = requests.post(url, json=payload, timeout=15)
    print("Login status:", r.status_code)
    print("Login response:", r.text[:500])

    r.raise_for_status()
    data = r.json()

    return data["data"]["token"]


def search_movie(token, query):
    url = f"{BASE_URL}/search"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    params = {
        "query": query,
        "type": "movie"
    }

    r = requests.get(url, headers=headers, params=params, timeout=15)
    print("\nSearch status:", r.status_code)
    print("Search response preview:")
    print(r.text[:1500])

    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    token = login()
    print("\nTOKEN WORKS")

    result = search_movie(token, "The Dark Knight")

    print("\nParsed JSON keys:")
    print(result.keys())

    print("\nFirst result:")
    if result.get("data"):
        print(json.dumps(result["data"][0], indent=2)[:2000])
    else:
        print("No data returned")