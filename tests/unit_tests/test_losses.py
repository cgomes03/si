import os
from unittest import TestCase

import numpy as np
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from datasets import DATASETS_PATH
from si.neural_networks.losses import BinaryCrossEntropy, CategoricalCrossEntropy, LossFunction, MeanSquaredError


class TestLosses(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_mean_squared_error_loss(self):

        error = MeanSquaredError().loss(self.dataset.y, self.dataset.y)

        self.assertEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = MeanSquaredError().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_binary_cross_entropy_loss(self):

        error = BinaryCrossEntropy().loss(self.dataset.y, self.dataset.y)

        self.assertAlmostEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = BinaryCrossEntropy().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])



class TestCategoricalCrossEntropy(TestCase):
    """
    Unit tests for CategoricalCrossEntropy loss function.    
    """

    def setUp(self):
        """Set up test fixtures with one-hot encoded data."""
        self.n_samples = 50
        self.n_classes = 3
        
        # Create one-hot encoded true labels
        self.y_true_indices = np.random.randint(0, self.n_classes, self.n_samples)
        self.y_true = np.eye(self.n_classes)[self.y_true_indices]
        
        # Create softmax predictions (random probabilities that sum to 1)
        logits = np.random.randn(self.n_samples, self.n_classes)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def test_categorical_cross_entropy_loss_perfect_prediction(self):
        """Test that loss is near zero when predictions equal true labels."""
        loss = CategoricalCrossEntropy().loss(self.y_true, self.y_true)
        self.assertAlmostEqual(loss, 0, places=5)

    def test_categorical_cross_entropy_loss_shape(self):
        """Test that loss returns a scalar value."""
        loss = CategoricalCrossEntropy().loss(self.y_true, self.y_pred)
        self.assertIsInstance(loss, (float, np.floating))

    def test_categorical_cross_entropy_loss_positive(self):
        """Test that loss is always non-negative."""
        loss = CategoricalCrossEntropy().loss(self.y_true, self.y_pred)
        self.assertGreaterEqual(loss, 0)

    def test_categorical_cross_entropy_derivative_shape(self):
        """Test that derivative has the correct shape."""
        derivative = CategoricalCrossEntropy().derivative(self.y_true, self.y_pred)
        self.assertEqual(derivative.shape, self.y_true.shape)

    def test_categorical_cross_entropy_derivative_formula(self):
        """
        Test the derivative formula from slide: ∂L/∂Y = -Y / Y*
        where Y = y_true and Y* = y_pred
        """
        derivative = CategoricalCrossEntropy().derivative(self.y_true, self.y_pred)
        
        # Expected: -y_true / y_pred (with clipping)
        p = np.clip(self.y_pred, 1e-15, 1 - 1e-15)
        expected_derivative = -(self.y_true / p)
        
        np.testing.assert_array_almost_equal(derivative, expected_derivative)