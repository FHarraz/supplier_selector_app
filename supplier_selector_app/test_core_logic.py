import unittest
import pandas as pd
import numpy as np
from app import topsis_score

class TestTopsisLogic(unittest.TestCase):

    def setUp(self):
        # Sample mini dataset with corrected column name
        self.df = pd.DataFrame({
            "Sales per Unit": [10, 20, 15],
            "Quantity": [5, 10, 7],
            "Rating": [4.5, 4.8, 4.6],
            "Sentiment": [0.7, 0.9, 0.8]  # Capital 'S' matches app.py
        })
        self.equal_weights = np.array([0.25, 0.25, 0.25, 0.25])

    def test_score_range(self):
        scores = topsis_score(self.df, self.equal_weights)
        self.assertTrue(((scores >= 0) & (scores <= 100)).all(), "Scores should be between 0 and 100")

    def test_score_ranking_order(self):
        scores = topsis_score(self.df, self.equal_weights)
        ranked = scores.argsort()[::-1]
        self.assertEqual(ranked[0], 1, "Second row should have highest score")

    def test_score_type(self):
        scores = topsis_score(self.df, self.equal_weights)
        self.assertIsInstance(scores, np.ndarray, "Scores should be returned as a numpy array")

if __name__ == '__main__':
    unittest.main()