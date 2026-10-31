"""Hybrid movie recommender and personalized natural-language explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import BPRMatrixFactorization, TextAndIntentModel, minmax
from .nlp_features import enrich_movies


class MovieRecommender:
    def __init__(self, factors=32, bpr_epochs=18, n_topics=7, seed=42):
        self.factors, self.bpr_epochs, self.n_topics, self.seed = factors, bpr_epochs, n_topics, seed

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies = enrich_movies(movies).reset_index(drop=True)
        self.ratings = ratings.copy().reset_index(drop=True)
        self.movie_ids = self.movies.movie_id.astype(int).tolist()
        self.movie_lookup = self.movies.set_index("movie_id", drop=False)
        self.text_model = TextAndIntentModel(n_topics=self.n_topics, seed=self.seed).fit(self.movies)
        self.bpr = BPRMatrixFactorization(factors=self.factors, epochs=self.bpr_epochs, seed=self.seed).fit(self.ratings)
        self.popularity = self.ratings.assign(positive=(self.ratings.rating >= 4.0).astype(int)).groupby("movie_id").positive.sum()
        self.ready = True
        return self

    def _user_dimensions(self, user_id: int) -> list[str]:
        history = self.ratings[(self.ratings.user_id == user_id) & (self.ratings.rating >= 4.0)]
        counts: dict[str, float] = {}
        for row in history.itertuples():
            movie = self.movie_lookup.loc[int(row.movie_id)] if int(row.movie_id) in self.movie_lookup.index else None
            if movie is None:
                continue
            for dim in str(movie.dimensions).split("|"):
                counts[dim] = counts.get(dim, 0.0) + max(0.1, float(row.rating) - 3.0)
        return [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)]

    def explain(self, user_id: int, movie_id: int) -> str:
        if movie_id not in self.movie_lookup.index:
            return "Recommended based on the available movie information."
        movie = self.movie_lookup.loc[movie_id]
        user_dims = self._user_dimensions(user_id)
        item_dims = [x for x in str(movie.dimensions).split("|") if x]
        overlap = [d for d in user_dims if d in item_dims]
        preferred = user_dims[:2]
        matching = overlap[:2] if overlap else item_dims[:2]
        if preferred and matching:
            return (
                f"Recommended because your highly rated movies often involve {', '.join(preferred)}. "
                f"This title also offers {', '.join(matching)}, alongside its {movie.genres} genre mix."
            )
        if matching:
            return f"This title is suggested for its {', '.join(matching)} dimensions and {movie.genres} genres."
        return f"This title is suggested from the learned user-item ranking profile; its genres are {movie.genres}."

    def recommend(self, user_id: int, k=10, algorithm="hybrid", exclude_seen=True) -> pd.DataFrame:
        if not getattr(self, "ready", False):
            raise RuntimeError("Call fit(movies, ratings) before recommend().")
        user_id = int(user_id)
        seen = set(self.ratings.loc[self.ratings.user_id == user_id, "movie_id"].astype(int))
        candidates = [m for m in self.movie_ids if not (exclude_seen and m in seen)]
        if not candidates:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "score", "explanation"])
        bpr_scores = self.bpr.score(user_id, candidates)
        content_scores = self.text_model.content_scores(user_id, candidates, self.ratings)
        intent_scores = self.text_model.intent_scores(user_id, candidates, self.ratings)
        popularity_scores = np.asarray([float(self.popularity.get(m, 0)) for m in candidates])
        methods = {
            "hybrid": 0.20 * minmax(bpr_scores) + 0.65 * minmax(content_scores) + 0.15 * minmax(intent_scores),
            "bpr": minmax(bpr_scores),
            "content": minmax(content_scores),
            "intent": minmax(intent_scores),
            "popularity": minmax(popularity_scores),
        }
        if algorithm not in methods:
            raise ValueError(f"Unknown algorithm {algorithm!r}. Choose from {sorted(methods)}")
        scores = methods[algorithm]
        order = np.argsort(-scores)[:int(k)]
        rows = []
        for idx in order:
            mid = int(candidates[idx])
            movie = self.movie_lookup.loc[mid]
            rows.append({
                "movie_id": mid,
                "title": str(movie.title),
                "genres": str(movie.genres),
                "score": float(scores[idx]),
                "explanation": self.explain(user_id, mid),
                "dimensions": str(movie.dimensions),
            })
        return pd.DataFrame(rows)
