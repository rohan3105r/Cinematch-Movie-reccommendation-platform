"""Small offline smoke tests; run with python -m unittest discover -s tests."""
import unittest

from src.data import load_data
from src.nlp_features import extract_dimensions, enrich_movies
from src.recommender import MovieRecommender
from src.evaluation import evaluate


class MovieRecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies, cls.ratings = load_data()

    def test_dimension_extraction(self):
        dims = extract_dimensions("A funny story with a memorable soundtrack and emotional characters", "Comedy|Drama")
        self.assertIn("humour and comedy", dims)
        self.assertIn("music and soundtrack", dims)
        self.assertIn("emotion and relationships", dims)

    def test_enrichment_adds_combined_text(self):
        enriched = enrich_movies(self.movies.head(2))
        self.assertIn("combined_text", enriched.columns)
        self.assertTrue(enriched.combined_text.str.len().gt(0).all())

    def test_recommender_returns_ranked_explanations(self):
        model = MovieRecommender(bpr_epochs=3).fit(self.movies, self.ratings)
        user_id = int(self.ratings.user_id.iloc[0])
        result = model.recommend(user_id, k=5)
        self.assertLessEqual(len(result), 5)
        self.assertIn("explanation", result.columns)
        self.assertTrue(result.score.is_monotonic_decreasing)

    def test_evaluation_has_expected_methods_and_range(self):
        metrics, per_user, _model, _train, _test = evaluate(self.movies, self.ratings)
        self.assertEqual(set(metrics.Method), {"popularity", "content", "bpr", "hybrid"})
        self.assertTrue(metrics["HR@10"].between(0, 1).all())
        self.assertGreater(len(per_user), 0)


if __name__ == "__main__":
    unittest.main()
