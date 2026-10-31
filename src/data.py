"""Dataset loading and leave-one-positive-out data splitting."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "demo"


def load_data(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    movies_path, ratings_path = folder / "movies.csv", folder / "ratings.csv"
    if not movies_path.exists() or not ratings_path.exists():
        raise FileNotFoundError(
            f"Expected movies.csv and ratings.csv in {folder}. "
            "Run `python scripts/generate_demo_data.py` or the MovieLens downloader."
        )
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    required_movies = {"movie_id", "title", "genres", "overview"}
    required_ratings = {"user_id", "movie_id", "rating"}
    if not required_movies.issubset(movies.columns):
        raise ValueError(f"movies.csv must contain {sorted(required_movies)}")
    if not required_ratings.issubset(ratings.columns):
        raise ValueError(f"ratings.csv must contain {sorted(required_ratings)}")
    movies["movie_id"] = movies["movie_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating", "user_id", "movie_id"])
    if "timestamp" not in ratings.columns:
        ratings["timestamp"] = range(len(ratings))
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce").fillna(0)
    return movies, ratings


def leave_one_positive_out(ratings: pd.DataFrame, positive_threshold: float = 4.0):
    """Hold out each eligible user's latest positive interaction.

    This is a simple reproducible ranking split. Unobserved items are treated
    as candidates, not as verified negatives; that assumption is documented.
    """
    test_rows = []
    for user_id, group in ratings.groupby("user_id"):
        positives = group[group["rating"] >= positive_threshold]
        if len(positives) < 2 or len(group) < 5:
            continue
        held_out = positives.sort_values(["timestamp", "movie_id"]).iloc[-1]
        test_rows.append(held_out.name)
    test = ratings.loc[test_rows].copy() if test_rows else ratings.iloc[0:0].copy()
    train = ratings.drop(index=test_rows).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)
