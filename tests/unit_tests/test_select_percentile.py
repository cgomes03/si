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
        """Verify that fit correctly calculates F and p values for each feature."""
        select_percentile = SelectPercentile(score_func=f_classification, percentile=40)
        select_percentile.fit(self.dataset)
        
        # F and p must have shape equal to the number of features
        self.assertEqual(select_percentile.F.shape[0], self.dataset.X.shape[1])
        self.assertEqual(select_percentile.p.shape[0], self.dataset.X.shape[1])
        
        # They must not be None
        self.assertIsNotNone(select_percentile.F)
        self.assertIsNotNone(select_percentile.p)
        
        # F must have values > 0 (for f_classification)
        self.assertTrue(np.all(select_percentile.F > 0))
        
        # p must be between 0 and 1
        self.assertTrue(np.all(select_percentile.p >= 0))
        self.assertTrue(np.all(select_percentile.p <= 1))

    def test_transform_selects_top_features(self):
        """Verify that transform selects features with the highest scores."""
        percentile = 50
        select_percentile = SelectPercentile(score_func=f_classification, percentile=percentile)
        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)
        
        # Calculate expected number of features
        original_features = self.dataset.X.shape[1]
        expected_num_features = int(np.ceil(original_features * percentile / 100))
        
        self.assertEqual(new_dataset.X.shape[1], expected_num_features)
        
        # Get indices of selected features using isin
        selected_feature_indices = np.where(np.isin(self.dataset.features, new_dataset.features))[0]
        selected_feature_indices = np.sort(selected_feature_indices)
        
        # All selected features must be in the top percentile
        min_selected_score = np.min(select_percentile.F[selected_feature_indices])
        
        # Sort all scores and get the threshold
        sorted_indices = np.argsort(select_percentile.F)[-expected_num_features:]
        min_threshold_score = np.min(select_percentile.F[sorted_indices])
        
        self.assertGreaterEqual(min_selected_score, min_threshold_score)


    def test_transform_number_of_rows_unchanged(self):
        """Verify that the number of samples (rows) remains unchanged."""
        select_percentile = SelectPercentile(score_func=f_classification, percentile=50)
        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)
        
        self.assertEqual(new_dataset.X.shape[0], self.dataset.X.shape[0])

    def test_transform_percentile_100(self):
        """Verify that percentile=100 selects all features."""
        select_percentile = SelectPercentile(score_func=f_classification, percentile=100)
        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)
        
        # Must select all features
        self.assertEqual(new_dataset.X.shape[1], self.dataset.X.shape[1])
        np.testing.assert_array_equal(new_dataset.X, self.dataset.X)

    def test_transform_percentile_1(self):
        """Verify that percentile=1 selects at least 1 feature."""
        select_percentile = SelectPercentile(score_func=f_classification, percentile=1)
        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)
        
        # Must select at least 1 feature
        self.assertGreaterEqual(new_dataset.X.shape[1], 1)
        self.assertLess(new_dataset.X.shape[1], self.dataset.X.shape[1])

    def test_transform_features_list_updated(self):
        """Verify that the features list is correctly updated."""
        select_percentile = SelectPercentile(score_func=f_classification, percentile=50)
        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)
        
        # The number of features in the list must match the number of columns
        self.assertEqual(len(new_dataset.features), new_dataset.X.shape[1])
        
        # All selected features must be from the original dataset
        for feature in new_dataset.features:
            self.assertIn(feature, self.dataset.features)

    def test_fit_and_transform_chain(self):
        """Verify that fit_transform works correctly."""
        percentile = 40
        select_percentile = SelectPercentile(score_func=f_classification, percentile=percentile)
        new_dataset = select_percentile.fit_transform(self.dataset)
        
        original_features = self.dataset.X.shape[1]
        expected_features = int(np.ceil(original_features * percentile / 100))
        
        self.assertEqual(new_dataset.X.shape[1], expected_features)
        self.assertEqual(new_dataset.X.shape[0], self.dataset.X.shape[0])


if __name__ == "__main__":
    unittest.main()
