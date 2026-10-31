"""Leave-one-positive-out ranking evaluation with common top-K metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import leave_one_positive_out
from .recommender import MovieRecommender
from .model import minmax

METHODS = ["popularity", "content", "bpr", "hybrid"]


def _rank_target(scores: np.ndarray, candidates: list[int], target: int) -> int:
    order = np.argsort(-np.asarray(scores, dtype=float))
    ranked = [candidates[i] for i in order]
    try:
        return ranked.index(int(target)) + 1
    except ValueError:
        return len(ranked) + 1


def evaluate(movies: pd.DataFrame, ratings: pd.DataFrame, seed=42):
    train, test = leave_one_positive_out(ratings)
    if len(test) == 0:
        raise ValueError("No eligible users for leave-one-positive-out evaluation.")
    model = MovieRecommender(seed=seed).fit(movies, train)
    movie_ids = movies.movie_id.astype(int).tolist()
    observations: dict[str, list[int]] = {m: [] for m in METHODS}
    per_user = []

    for row in test.itertuples(index=False):
        user_id, target = int(row.user_id), int(row.movie_id)
        seen = set(train.loc[train.user_id == user_id, "movie_id"].astype(int))
        candidates = [m for m in movie_ids if m not in seen]
        if target not in candidates or len(candidates) == 0:
            continue
        bpr = model.bpr.score(user_id, candidates)
        content = model.text_model.content_scores(user_id, candidates, train)
        intent = model.text_model.intent_scores(user_id, candidates, train)
        popularity = np.asarray([float(model.popularity.get(m, 0)) for m in candidates])
        score_map = {
            "popularity": minmax(popularity),
            "content": minmax(content),
            "bpr": minmax(bpr),
            "hybrid": 0.20 * minmax(bpr) + 0.65 * minmax(content) + 0.15 * minmax(intent),
        }
        row_metrics = {"user_id": user_id, "target_movie_id": target, "candidate_count": len(candidates)}
        for method, scores in score_map.items():
            rank = _rank_target(scores, candidates, target)
            observations[method].append(rank)
            row_metrics[f"{method}_rank"] = rank
        per_user.append(row_metrics)

    if not per_user:
        raise ValueError("No users remained after candidate construction.")
    metric_rows = []
    for method in METHODS:
        ranks = np.asarray(observations[method], dtype=int)
        n = len(ranks)
        metric_rows.append({
            "Method": method,
            "HR@5": float(np.mean(ranks <= 5)),
            "HR@10": float(np.mean(ranks <= 10)),
            "Recall@5": float(np.mean(ranks <= 5)),
            "Recall@10": float(np.mean(ranks <= 10)),
            "NDCG@5": float(np.mean([1 / np.log2(r + 1) if r <= 5 else 0.0 for r in ranks])),
            "NDCG@10": float(np.mean([1 / np.log2(r + 1) if r <= 10 else 0.0 for r in ranks])),
            "MAP@10": float(np.mean([1 / r if r <= 10 else 0.0 for r in ranks])),
            "Precision@10": float(np.mean([1 / 10 if r <= 10 else 0.0 for r in ranks])),
            "Evaluated users": n,
        })
    metrics_df = pd.DataFrame(metric_rows)
    return metrics_df, pd.DataFrame(per_user), model, train, test
