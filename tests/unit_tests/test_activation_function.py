from unittest import TestCase

from datasets import DATASETS_PATH

import os
import numpy as np
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.neural_networks.activation import ReLUActivation, SigmoidActivation, TanhActivation, SoftmaxActivation

class TestSigmoidLayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        sigmoid_layer = SigmoidActivation()
        result = sigmoid_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = SigmoidActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])


class TestRELULayer(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):

        relu_layer = ReLUActivation()
        result = relu_layer.activation_function(self.dataset.X)
        self.assertTrue(all([i >= 0 for j in range(result.shape[1]) for i in result[:, j]]))


    def test_derivative(self):
        sigmoid_layer = ReLUActivation()
        derivative = sigmoid_layer.derivative(self.dataset.X)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])




class TestTanhLayer(TestCase):
    """
    Unit tests for TanhActivation layer.    
    """

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):
        """Test that tanh output is in range [-1, 1]"""
        tanh_layer = TanhActivation()
        result = tanh_layer.activation_function(self.dataset.X)
        
        # Check shape
        self.assertEqual(result.shape, self.dataset.X.shape)
        
        # Check that all values are in range [-1, 1]
        self.assertTrue(all([i >= -1 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))

    def test_derivative(self):
        """Test that tanh derivative has correct shape"""
        tanh_layer = TanhActivation()
        derivative = tanh_layer.derivative(self.dataset.X)
        
        # Check shape
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])
        
        # Check that all derivative values are in range [0, 1]
        # Since derivative = 1 - tanh²(x), max is 1 (at x=0), min is 0
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(derivative.shape[1]) for i in derivative[:, j]]))


class TestSoftmaxLayer(TestCase):
    """
    Unit tests for SoftmaxActivation layer.    
    """

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_activation_function(self):
        """Test that softmax output is probability distribution"""
        softmax_layer = SoftmaxActivation()
        result = softmax_layer.activation_function(self.dataset.X)
        
        # Check shape
        self.assertEqual(result.shape, self.dataset.X.shape)
        
        # Check that all values are in range [0, 1] (probabilities)
        self.assertTrue(all([i >= 0 and i <= 1 for j in range(result.shape[1]) for i in result[:, j]]))
        
        # Check that each row sums to 1 (probability distribution)
        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(result.shape[0]), decimal=5)

    def test_derivative(self):
        """Test that softmax derivative has correct shape"""
        softmax_layer = SoftmaxActivation()
        derivative = softmax_layer.derivative(self.dataset.X)
        
        # Check shape (for batch, it's simplified to element-wise)
        self.assertEqual(derivative.shape[0], self.dataset.X.shape[0])
        self.assertEqual(derivative.shape[1], self.dataset.X.shape[1])
