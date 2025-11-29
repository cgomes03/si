import unittest
import numpy as np
from si.data.dataset import Dataset
from si.decomposition.pca import PCA

class TestPCA(unittest.TestCase):
    def setUp(self):
        # Col 2 = Col 1 + 1; Col 3 = Col 2 + 1.
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]], dtype=float)
        y = np.array([0, 1, 0, 1])
        self.dataset = Dataset(X, y, features=['F1', 'F2', 'F3'], label='y')

    def test_fit(self):
        pca = PCA(n_components=2)
        pca.fit(self.dataset)

        # 1. Check Mean (Calculado dinamicamente)
        expected_mean = np.mean(self.dataset.X, axis=0)
        np.testing.assert_array_almost_equal(pca.mean, expected_mean)

        # 2. Check Components Dimensions
        self.assertEqual(pca.components.shape, (2, 3))
        
        # 3. Check Explained Variance
        self.assertAlmostEqual(pca.explained_variance[0], 1.0, places=5)
        self.assertAlmostEqual(pca.explained_variance[1], 0.0, places=5)

    def test_transform(self):
        pca = PCA(n_components=2)
        pca.fit(self.dataset)
        reduced_ds = pca.transform(self.dataset)

        # 1. Check Output Shape
        self.assertEqual(reduced_ds.X.shape, (4, 2))
        
        # 2. Check Labels Preserved
        np.testing.assert_array_equal(reduced_ds.y, self.dataset.y)
        
        # 3. Check Orthogonality
        dot_product = np.dot(pca.components[0], pca.components[1])
        self.assertAlmostEqual(dot_product, 0.0, places=10)

if __name__ == "__main__":
    unittest.main()