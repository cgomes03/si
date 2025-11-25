from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH
import os

from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_stratified_train_test_split(self):
        # 1. execute the split
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        
        # 2. verify sizes
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)
        
        # 3. verify class proportions in the test set

        classes, total_counts = np.unique(self.dataset.y, return_counts=True)
        _, test_counts = np.unique(test.y, return_counts=True)

        
        self.assertEqual(len(classes), len(test_counts))

        expected_test_counts = (total_counts * 0.2).astype(int)

        for c_total, c_test, c_expected in zip(total_counts, test_counts, expected_test_counts):
            self.assertIn(c_test, {np.floor(c_total * 0.2), np.ceil(c_total * 0.2)})
            self.assertEqual(c_test, c_expected)        