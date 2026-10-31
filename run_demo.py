"""Print a short recommendation demo in the terminal."""
from pathlib import Path
from src.data import load_data
from src.recommender import MovieRecommender


def main():
    movies, ratings = load_data()
    model = MovieRecommender().fit(movies, ratings)
    user_id = int(sorted(ratings.user_id.unique())[0])
    print(f"CineMatch demo - User {user_id}\n")
    result = model.recommend(user_id, k=8, algorithm="hybrid")
    for rank, row in enumerate(result.itertuples(index=False), start=1):
        print(f"{rank}. {row.title} ({row.genres})  score={row.score:.3f}")
        print(f"   Why: {row.explanation}\n")
    out = Path(__file__).resolve().parent / "results" / "sample_recommendations.csv"
    result.to_csv(out, index=False)
    print(f"Saved recommendations to {out}")


if __name__ == "__main__":
    main()
