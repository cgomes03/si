import unittest
import os
import numpy as np
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification

class TestSelectPercentile(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join('datasets', 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        select_percentile = SelectPercentile(score_func=f_classification, percentile=40)
        select_percentile.fit(self.dataset)
        
        self.assertTrue(select_percentile.F.shape[0] > 0)
        self.assertTrue(select_percentile.p.shape[0] > 0)
        self.assertEqual(select_percentile.F.shape[0], self.dataset.X.shape[1])
        self.assertEqual(select_percentile.p.shape[0], self.dataset.X.shape[1])

    def test_transform(self):
        percentile = 40
        select_percentile = SelectPercentile(score_func=f_classification, percentile=percentile)
        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)

        original_features = self.dataset.X.shape[1]
        
        # Replicate the logic: (Total Features * Percentile) / 100
        expected_features = int(np.ceil(original_features * percentile / 100))
        
        # Ensure at least 1 feature is selected
        expected_features = max(1, expected_features)
        
        self.assertEqual(new_dataset.X.shape[1], expected_features)
        
        # Verify strict reduction
        self.assertLess(len(new_dataset.features), len(self.dataset.features))
        self.assertLess(new_dataset.X.shape[1], self.dataset.X.shape[1])
        
        self.assertEqual(new_dataset.X.shape[0], self.dataset.X.shape[0])

if __name__ == "__main__":
    unittest.main()