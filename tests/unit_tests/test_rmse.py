import unittest
import numpy as np
from sklearn.metrics import root_mean_squared_error
from si.metrics.rmse import rmse


class TestRMSE(unittest.TestCase):
    def test_rmse(self):
        y_true = np.array([2, 4, 6, 8])
        y_pred = np.array([2.5, 3.5, 6.5, 7.5])
        
        expected_rmse = root_mean_squared_error(y_true, y_pred)
        result = rmse(y_true, y_pred)
        
        self.assertEqual(result, expected_rmse)

if __name__ == '__main__':
    unittest.main()