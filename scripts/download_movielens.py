"""Optional downloader/converter for the public MovieLens 100K dataset.

Run: python scripts/download_movielens.py
The script downloads the GroupLens ml-100k archive and creates data/ml-100k.
Internet access is required. This is optional; the demo app is offline-capable.
"""
from __future__ import annotations
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ml-100k"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
GENRE_COLS = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urlopen(URL, timeout=60) as response:
        payload = response.read()
    with ZipFile(BytesIO(payload)) as archive:
        movie_raw = pd.read_csv(archive.open("ml-100k/u.item"), sep="|", encoding="latin-1", header=None)
        rating_raw = pd.read_csv(archive.open("ml-100k/u.data"), sep="\t", header=None, names=["user_id", "movie_id", "rating", "timestamp"])
    # u.item columns: id, title, dates, URL, then 19 binary genre indicators.
    movie_raw = movie_raw.iloc[:, :5 + len(GENRE_COLS)].copy()
    movie_raw.columns = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLS
    movie_rows = []
    for row in movie_raw.itertuples(index=False, name=None):
        genres = [g for g, flag in zip(GENRE_COLS, row[5:]) if int(flag) == 1]
        # Build compact textual input from available MovieLens metadata. ml-100k
        # has genres but does not supply plot synopses/reviews.
        genres = [g for g in genres if g != "unknown"] or ["Drama"]
        overview = "A movie combining " + ", ".join(genres) + " storytelling and themes."
        movie_rows.append({"movie_id": int(row[0]), "title": str(row[1]),
                           "genres": "|".join(genres), "overview": overview, "tags": ""})
    pd.DataFrame(movie_rows).to_csv(OUT / "movies.csv", index=False)
    rating_raw.to_csv(OUT / "ratings.csv", index=False)
    print(f"Saved {len(movie_rows)} movies and {len(rating_raw)} ratings to {OUT}")
    print("Note: MovieLens 100K supplies genres/ratings; this project synthesizes short genre descriptions because plot text is not part of u.item.")


if __name__ == "__main__":
    main()
