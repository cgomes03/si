import unittest


import numpy as np


from si.data.dataset import Dataset




class TestDataset(unittest.TestCase):


    def setUp(self):
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        y = np.array([1, 2])


        features = np.array(['a', 'b', 'c'])
        label = 'y'
        self.dataset = Dataset(X, y, features, label)



    def test_dataset_construction(self):


        self.assertEqual(2.5, self.dataset.get_mean()[0])
        self.assertEqual((2, 3), self.dataset.shape())
        self.assertTrue(self.dataset.has_label())
        self.assertEqual(1, self.dataset.get_classes()[0])
        self.assertEqual(2.25, self.dataset.get_variance()[0])
        self.assertEqual(1, self.dataset.get_min()[0])
        self.assertEqual(4, self.dataset.get_max()[0])
        self.assertEqual(2.5, self.dataset.summary().iloc[0, 0])


    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())



    def test_dropna(self):
        self.dataset.X[0, 0] = np.nan
        resultado = self.dataset.dropna()
        self.assertTrue(np.all(~np.isnan(resultado.X)))


    def test_fillna_median(self):
        X_test = self.dataset.X.copy()
        X_test[0, 0] = np.nan
        resultado = self.dataset.fillna(strategy="median")
        self.assertTrue(np.all(~np.isnan(resultado.X)))


    def test_fillna_mean(self):
        X_test = self.dataset.X.copy()
        X_test[1, 1] = np.nan
        resultado = self.dataset.fillna(strategy="mean")
        self.assertTrue(np.all(~np.isnan(resultado.X)))


    def test_remove_by_index(self):
        index = 0
        resultado = self.dataset.remove_by_index(index)
        self.assertEqual(resultado.shape()[0], self.dataset.X.shape[0] - 1)


if __name__ == "__main__":
    unittest.main()