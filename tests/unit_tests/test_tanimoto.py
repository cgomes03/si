import unittest
import numpy as np

from si.statistics.tanimoto_similarity import tanimoto_similarity


class TestTanimotoSimilarity(unittest.TestCase):
    
    def test_tanimoto_similarity(self):
        # Test data (binary)
        x = np.array([1, 0, 1, 1])
        y = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        
        # Execute your function
        our_similarity = tanimoto_similarity(x, y)
        
        expected_similarity = np.array([0.66666667, 0.25])
        
        # Verification
        assert np.allclose(our_similarity, expected_similarity)

if __name__ == "__main__":
    unittest.main()