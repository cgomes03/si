import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    
    def test_dropna(self):
        # Create a local dataset
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
        y = np.array([1, 2, 1, 2])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        # Set a NaN
        dataset.X[0, 0] = np.nan
        
        # Calculate expected shape dynamically
        expected_rows = X.shape[0] - 1
        
        dataset = dataset.dropna()
        
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(dataset.shape()[0], expected_rows)

    def test_fillna_median(self):
        # Create dataset
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
        y = np.array([1, 2, 1, 2])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        # Insert NaN at specific position
        nan_row, nan_col = 0, 0
        dataset.X[nan_row, nan_col] = np.nan
        
        col_values = dataset.X[:, nan_col]
        expected_value = np.nanmedian(col_values)

        dataset = dataset.fillna(value="median")
        
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(dataset.X[nan_row, nan_col], expected_value)

    def test_fillna_mean(self):
        # Create dataset
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
        y = np.array([1, 2, 1, 2])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        # Insert NaN
        nan_row, nan_col = 1, 1
        dataset.X[nan_row, nan_col] = np.nan
        
        col_values = dataset.X[:, nan_col]
        expected_value = np.nanmean(col_values)
        
        dataset = dataset.fillna(value="mean")
        
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(dataset.X[nan_row, nan_col], expected_value)

    def test_remove_by_index(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])
        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)
        
        index = 0
        original_shape = dataset.shape()[0]
        
        # We expect the new first row to be what was previously the second row
        expected_first_row = dataset.X[1].copy()
        
        dataset = dataset.remove_by_index(index)
        
        self.assertEqual(dataset.shape()[0], original_shape - 1)
        np.testing.assert_array_equal(dataset.X[0], expected_first_row)


if __name__ == "__main__":
    unittest.main()