from unittest import TestCase

import numpy as np
import os

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares


class TestRidgeRegressionLeastSquares(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        model.fit(self.dataset)

        # Verifica se os parâmetros foram estimados
        self.assertIsNotNone(model.theta)
        self.assertIsNotNone(model.theta_zero)
        self.assertIsNotNone(model.mean)
        self.assertIsNotNone(model.std)

        # Verifica se theta tem o tamanho correto (número de features)
        self.assertEqual(len(model.theta), self.dataset.shape()[1])

    def test_predict(self):
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        model.fit(self.dataset)

        predictions = model.predict(self.dataset)

        # Verifica se o número de previsões é igual ao número de samples
        self.assertEqual(len(predictions), self.dataset.shape()[0])

    def test_score(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)

        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        model.fit(train)

        score = model.score(test)

        # Verifica se o score (MSE) é um número não negativo
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_score_without_scale(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)

        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=False)
        model.fit(train)

        score = model.score(test)

        # Verifica se o score (MSE) é um número não negativo
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)

    def test_different_l2_penalties(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)

        scores = []
        for l2 in [0.1, 1.0, 10.0]:
            model = RidgeRegressionLeastSquares(l2_penalty=l2, scale=True)
            model.fit(train)
            scores.append(model.score(test))

        # Verifica se todos os scores são válidos
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
