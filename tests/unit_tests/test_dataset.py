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
        X = np.array([[1,  2,  3],
                    [4,  5,  6],
                    [7,  8,  9],
                    [10, 11, 12]], dtype=float)
        y = np.array([1, 2, 1, 2])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        # Set a NaN 
        nan_row = 0
        dataset.X[nan_row, 0] = np.nan

        # Expected: remove the row with NaN
        expected_X = X[1:, :].copy()
        expected_y = y[1:].copy()
        expected_rows = X.shape[0] - 1

        dataset = dataset.dropna()

        # X should have no NaNs
        self.assertFalse(np.isnan(dataset.X).any())
        # Number of rows should be reduced by 1
        self.assertEqual(dataset.shape()[0], expected_rows)
        # X should match expected
        np.testing.assert_array_equal(dataset.X, expected_X)
        # y must match expected
        np.testing.assert_array_equal(dataset.y, expected_y)
        

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

    def test_fillna_scalar_value(self):
        # Test filling NaNs with a scalar value
        X = np.array([[1,  np.nan, 3],
                    [4,  5, 6],
                    [np.nan, 8,  9]], dtype=float)
        y = np.array([1, 2, 1])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        fill_value = 0.0
        dataset = dataset.fillna(value=fill_value)

        
        self.assertFalse(np.isnan(dataset.X).any())
        # the nan positions should now be equal to fill_value
        self.assertEqual(dataset.X[0, 1], fill_value)
        self.assertEqual(dataset.X[2, 0], fill_value)


    def test_fillna_invalid_value_raises(self):
        # validate input
        X = np.array([[1, np.nan]], dtype=float)
        y = np.array([0])
        dataset = Dataset(X, y, features=['a', 'b'], label='y')

        # if invalid string
        with self.assertRaises(ValueError):
            dataset.fillna(value="foo")

        # if invalid list
        with self.assertRaises(ValueError):
            dataset.fillna(value=["mean"])


    def test_remove_by_index(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([1, 2, 3])
        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)
        
        # Test 1: Remove first index (valid)
        original_shape = dataset.shape()[0]
        expected_first_row = dataset.X[1].copy()
        dataset = dataset.remove_by_index(0)
        
        self.assertEqual(dataset.shape()[0], original_shape - 1)
        np.testing.assert_array_equal(dataset.X[0], expected_first_row)
        
        # Test 2: Remove last index (valid)
        dataset = Dataset(X, y, features, label)  # Reset dataset
        original_shape = dataset.shape()[0]
        expected_last_row = dataset.X[1].copy()
        dataset = dataset.remove_by_index(2)
        
        self.assertEqual(dataset.shape()[0], original_shape - 1)
        np.testing.assert_array_equal(dataset.X[-1], expected_last_row)
        
        # Test 3: Index out of range (invalid)
        dataset = Dataset(X, y, features, label)  # Reset Dataset
        with self.assertRaises(IndexError):  
            dataset.remove_by_index(10)

if __name__ == "__main__":
    unittest.main()