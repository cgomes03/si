import unittest
import numpy as np
from si.neural_networks.layers import Dropout
from si.neural_networks.optimizers import Optimizer


class MockOptimizer(Optimizer):
    """Mock optimizer for testing purposes."""
    
    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate)

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        return w - self.learning_rate * grad_loss_w


class TestDropout(unittest.TestCase):
    """
    Unit tests for the Dropout layer.  
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_shape = (100, 50)
        self.input_data = np.random.randn(*self.input_shape)
        self.optimizer = MockOptimizer()
    
    def test_dropout_initialization(self):
        """Test Dropout layer initialization with valid and invalid probabilities."""
        # Valid probability
        dropout = Dropout(probability=0.5)
        self.assertEqual(dropout.probability, 0.5)
        
        # Invalid probabilities should raise ValueError
        with self.assertRaises(ValueError):
            Dropout(probability=0.0)
        with self.assertRaises(ValueError):
            Dropout(probability=1.0)
    
    def test_dropout_parameters(self):
        """Test that Dropout layer has 0 parameters."""
        dropout = Dropout(probability=0.5)
        self.assertEqual(dropout.parameters(), 0)
    
    def test_dropout_output_shape(self):
        """Test that Dropout preserves input shape."""
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape(self.input_shape)
        self.assertEqual(dropout.output_shape(), self.input_shape)
    
    def test_forward_propagation_training(self):
        """
        Test forward propagation during training.
        Should dropout approximately 50% of values with probability=0.5.
        """
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape(self.input_shape)
        dropout.initialize(self.optimizer)
        
        output = dropout.forward_propagation(self.input_data, training=True)
        
        # Check shape is preserved
        self.assertEqual(output.shape, self.input_data.shape)
        
        # Check that approximately 50% of values are zero
        zero_count = np.sum(output == 0)
        expected_zeros = self.input_data.size * 0.5
        # Allow 20% tolerance
        self.assertAlmostEqual(zero_count, expected_zeros, delta=self.input_data.size * 0.2)
    
    def test_forward_propagation_inference(self):
        """
        Test forward propagation during inference.
        Should return input unchanged.
        """
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape(self.input_shape)
        dropout.initialize(self.optimizer)
        
        output = dropout.forward_propagation(self.input_data, training=False)
        
        # Output should equal input exactly
        np.testing.assert_array_equal(output, self.input_data)
    
    def test_backward_propagation(self):
        """
        Test backward propagation.
        Should apply mask to output_error.
        """
        dropout = Dropout(probability=0.5)
        dropout.set_input_shape(self.input_shape)
        dropout.initialize(self.optimizer)
        
        # Forward pass to generate mask
        dropout.forward_propagation(self.input_data, training=True)
        
        # Create output error
        output_error = np.random.randn(*self.input_shape)
        
        # Backward pass
        input_error = dropout.backward_propagation(output_error)
        
        # Check shape is preserved
        self.assertEqual(input_error.shape, self.input_shape)
        
        # Check that error is zero where mask is zero
        np.testing.assert_array_equal(input_error[dropout.mask == 0], 0)


if __name__ == '__main__':
    unittest.main()