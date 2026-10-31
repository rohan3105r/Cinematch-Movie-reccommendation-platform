"""BPR factorization and latent-topic preference models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


class BPRMatrixFactorization:
    """Lightweight Bayesian Personalized Ranking matrix factorization.

    Pairwise updates optimize -log(sigmoid(score(u,i)-score(u,j))) for a
    positive item i and a sampled negative item j.
    """
    def __init__(self, factors=32, epochs=18, learning_rate=0.025, reg=0.01, seed=42):
        self.factors = int(factors)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.reg = float(reg)
        self.seed = int(seed)

    def fit(self, ratings: pd.DataFrame):
        rng = np.random.default_rng(self.seed)
        self.users = sorted(ratings.user_id.unique().astype(int).tolist())
        self.items = sorted(ratings.movie_id.unique().astype(int).tolist())
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.item_to_idx = {m: i for i, m in enumerate(self.items)}
        n_users, n_items = len(self.users), len(self.items)
        self.P = rng.normal(0, 0.08, (n_users, self.factors))
        self.Q = rng.normal(0, 0.08, (n_items, self.factors))
        self.item_bias = np.zeros(n_items, dtype=float)

        positive_pairs: dict[int, list[int]] = {u: [] for u in self.users}
        negative_pairs: dict[int, list[int]] = {u: [] for u in self.users}
        seen: dict[int, set[int]] = {u: set() for u in self.users}
        for row in ratings.itertuples(index=False):
            u, m, rating = int(row.user_id), int(row.movie_id), float(row.rating)
            if m not in self.item_to_idx:
                continue
            seen[u].add(self.item_to_idx[m])
            if rating >= 4.0:
                positive_pairs[u].append(self.item_to_idx[m])
            elif rating <= 2.0:
                negative_pairs[u].append(self.item_to_idx[m])
        all_items = np.arange(n_items)
        self.global_mean = float(ratings.rating.mean()) if len(ratings) else 3.0
        samples = [(u, i) for u, arr in positive_pairs.items() for i in arr]
        if not samples:
            self.ready = True
            return self

        for _epoch in range(self.epochs):
            rng.shuffle(samples)
            for u, pos_i in samples:
                user_idx = self.user_to_idx[u]
                neg_pool = negative_pairs[u]
                if neg_pool:
                    neg_j = int(rng.choice(neg_pool))
                else:
                    available = np.setdiff1d(all_items, np.array(list(seen[u]), dtype=int), assume_unique=False)
                    if len(available) == 0:
                        continue
                    neg_j = int(rng.choice(available))
                if pos_i == neg_j:
                    continue
                x = float(np.dot(self.P[user_idx], self.Q[pos_i] - self.Q[neg_j]) + self.item_bias[pos_i] - self.item_bias[neg_j])
                # Stable sigmoid(-x).
                grad = 1.0 / (1.0 + np.exp(np.clip(x, -30, 30)))
                pu = self.P[user_idx].copy()
                qi = self.Q[pos_i].copy()
                qj = self.Q[neg_j].copy()
                self.P[user_idx] += self.learning_rate * (grad * (qi - qj) - self.reg * pu)
                self.Q[pos_i] += self.learning_rate * (grad * pu - self.reg * qi)
                self.Q[neg_j] += self.learning_rate * (-grad * pu - self.reg * qj)
                self.item_bias[pos_i] += self.learning_rate * (grad - self.reg * self.item_bias[pos_i])
                self.item_bias[neg_j] += self.learning_rate * (-grad - self.reg * self.item_bias[neg_j])
        self.ready = True
        return self

    def score(self, user_id: int, movie_ids: list[int]) -> np.ndarray:
        if not getattr(self, "ready", False) or user_id not in self.user_to_idx:
            return np.zeros(len(movie_ids), dtype=float)
        uidx = self.user_to_idx[user_id]
        result = []
        for movie_id in movie_ids:
            midx = self.item_to_idx.get(int(movie_id))
            result.append(float(np.dot(self.P[uidx], self.Q[midx]) + self.item_bias[midx]) if midx is not None else 0.0)
        return np.asarray(result, dtype=float)


class TextAndIntentModel:
    """TF-IDF text profile plus NMF latent-topic profile with attention."""
    def __init__(self, n_topics=7, seed=42):
        self.n_topics = int(n_topics)
        self.seed = int(seed)

    def fit(self, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True).copy()
        self.movie_ids = self.movies.movie_id.astype(int).tolist()
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        self.tfidf = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1,
            max_features=12000, sublinear_tf=True
        )
        texts = self.movies["combined_text"].fillna("").astype(str).tolist()
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        self.tfidf_matrix = normalize(self.tfidf_matrix, norm="l2", axis=1)

        self.count_vectorizer = CountVectorizer(stop_words="english", max_features=2500, min_df=1)
        counts = self.count_vectorizer.fit_transform(texts)
        n_topics = max(2, min(self.n_topics, max(2, counts.shape[0] - 1), max(2, counts.shape[1] - 1)))
        self.nmf = NMF(n_components=n_topics, init="nndsvda", random_state=self.seed, max_iter=350)
        self.item_topics = self.nmf.fit_transform(counts)
        self.item_topics = normalize(self.item_topics, norm="l1", axis=1)
        self.topic_terms = self._topic_terms()
        return self

    def _topic_terms(self, top_n=5):
        words = np.asarray(self.count_vectorizer.get_feature_names_out())
        result = []
        for topic in self.nmf.components_:
            ids = np.argsort(topic)[::-1][:top_n]
            result.append([str(words[i]) for i in ids])
        return result

    def _positive_history(self, user_id: int, ratings: pd.DataFrame):
        user_ratings = ratings[(ratings.user_id == user_id) & (ratings.rating >= 4.0)]
        rows = [(self.movie_to_idx[int(r.movie_id)], float(r.rating - 3.0))
                for r in user_ratings.itertuples() if int(r.movie_id) in self.movie_to_idx]
        return rows

    def content_scores(self, user_id: int, movie_ids: list[int], ratings: pd.DataFrame) -> np.ndarray:
        history = self._positive_history(user_id, ratings)
        if not history:
            return np.zeros(len(movie_ids), dtype=float)
        idx = np.asarray([r[0] for r in history], dtype=int)
        weights = np.asarray([r[1] for r in history], dtype=float)
        profile = self.tfidf_matrix[idx].multiply(weights[:, None]).sum(axis=0)
        profile = normalize(np.asarray(profile), norm="l2")
        positions = [self.movie_to_idx.get(int(m), -1) for m in movie_ids]
        result = np.zeros(len(movie_ids), dtype=float)
        valid = [(i, p) for i, p in enumerate(positions) if p >= 0]
        if valid:
            item_indices = [p for _, p in valid]
            raw = np.asarray(self.tfidf_matrix[item_indices] @ profile.T).reshape(-1)
            for (out_idx, _), value in zip(valid, raw):
                result[out_idx] = float(value)
        return result

    def intent_scores(self, user_id: int, movie_ids: list[int], ratings: pd.DataFrame) -> np.ndarray:
        history = self._positive_history(user_id, ratings)
        if not history:
            return np.zeros(len(movie_ids), dtype=float)
        idx = np.asarray([r[0] for r in history], dtype=int)
        weights = np.asarray([r[1] for r in history], dtype=float)
        topic_profile = np.average(self.item_topics[idx], axis=0, weights=weights)
        # Attention over user latent intents: a softmax gives more weight to
        # topics with stronger mass in the user's positive interaction history.
        logits = topic_profile / 0.12
        logits = logits - np.max(logits)
        attention = np.exp(logits)
        attention = attention / (attention.sum() + 1e-12)
        positions = [self.movie_to_idx.get(int(m), -1) for m in movie_ids]
        scores = [float(np.dot(self.item_topics[p], attention)) if p >= 0 else 0.0 for p in positions]
        return np.asarray(scores, dtype=float)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    low, high = float(np.min(values)), float(np.max(values))
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)
