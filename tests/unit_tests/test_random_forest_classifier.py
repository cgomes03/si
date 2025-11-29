import unittest
import os
import numpy as np

from si.models.random_forest_classifier import RandomForestClassifier
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from datasets import DATASETS_PATH

class TestRandomForestClassifier(unittest.TestCase):

    def setUp(self):
        # Using the Iris dataset for testing
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit_predict_score(self):
        # Divide the dataset into training and testing sets
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.3, random_state=42)

        # Iniciate RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10, min_samples_split=2, max_depth=3, seed=42)
        
        # Trainning
        rf.fit(train_dataset)
        
        # Verify if the model has the correct number of trees
        self.assertEqual(len(rf.trees), 10)
        self.assertEqual(len(rf.trees[0]), 2) # Tuple (features, tree)
        
        # Preddiction and Scoring
        score = rf.score(test_dataset)
        print(f"Random Forest Accuracy: {score}")
        
        # having a reasonable accuracy
        self.assertGreater(score, 0.75)
        
        # Verify predictions shape
        predictions = rf.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.shape()[0])

if __name__ == '__main__':
    unittest.main()