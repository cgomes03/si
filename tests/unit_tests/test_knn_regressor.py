import unittest
import os
import numpy as np

from si.models.knn_regressor import KNNRegressor
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from datasets import DATASETS_PATH

class TestKNNRegressor(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit_predict_score(self):
        # Data Preparation
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)

        # Initialize KNN Regressor
        knn = KNNRegressor(k=3)
        
        # Training
        knn.fit(train_dataset)
        
        # Prediction
        predictions = knn.predict(test_dataset)
        
        # Verify the predictions shape
        self.assertEqual(predictions.shape[0], test_dataset.shape()[0])
        
        # Evaluation with RMSE
        score = knn.score(test_dataset)
        print(f"RMSE Score: {score}")
        self.assertTrue(score > 0)
        
        # Verify predictions are numeric
        self.assertTrue(np.issubdtype(predictions.dtype, np.floating) or np.issubdtype(predictions.dtype, np.integer))

if __name__ == '__main__':
    unittest.main()
