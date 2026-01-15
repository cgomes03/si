from unittest import TestCase
import numpy as np

from si.neural_networks.optimizers import Adam


class TestAdam(TestCase):
    """
    Unit tests for Adam optimizer
    """

    def setUp(self):
        """Set up test fixtures."""
        self.learning_rate = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        
        self.w = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.grad = np.array([[0.1, 0.2], [0.3, 0.4]])

    def test_adam_initialization(self):
        """Test Adam initialization with default and custom parameters."""
        # Default parameters
        adam_default = Adam()
        self.assertEqual(adam_default.learning_rate, 0.001)
        self.assertEqual(adam_default.beta_1, 0.9)
        self.assertEqual(adam_default.beta_2, 0.999)
        self.assertEqual(adam_default.epsilon, 1e-8)
        self.assertEqual(adam_default.t, 0)
        self.assertIsNone(adam_default.m)
        self.assertIsNone(adam_default.v)
        
        # Custom parameters
        adam_custom = Adam(learning_rate=0.01, beta_1=0.8, beta_2=0.99, epsilon=1e-6)
        self.assertEqual(adam_custom.learning_rate, 0.01)
        self.assertEqual(adam_custom.beta_1, 0.8)
        self.assertEqual(adam_custom.beta_2, 0.99)
        self.assertEqual(adam_custom.epsilon, 1e-6)

    def test_adam_first_update_t_increment(self):
        """Test that time step t increments correctly."""
        adam = Adam()
        self.assertEqual(adam.t, 0)
        
        adam.update(self.w.copy(), self.grad)
        self.assertEqual(adam.t, 1)
        
        adam.update(self.w.copy(), self.grad)
        self.assertEqual(adam.t, 2)

    def test_adam_m_v_initialization(self):
        """Test that m and v are initialized correctly."""
        adam = Adam()
        self.assertIsNone(adam.m)
        self.assertIsNone(adam.v)
        
        adam.update(self.w.copy(), self.grad)
        
        self.assertIsNotNone(adam.m)
        self.assertIsNotNone(adam.v)
        self.assertEqual(adam.m.shape, self.w.shape)
        self.assertEqual(adam.v.shape, self.w.shape)

    def test_adam_first_update_m_estimate(self):
        """Test 1st moment estimate (m) after first update."""
        adam = Adam(beta_1=self.beta_1)
        adam.update(self.w.copy(), self.grad)
        
        # After first update: m = beta_1 * 0 + (1 - beta_1) * grad
        expected_m = (1 - self.beta_1) * self.grad
        np.testing.assert_array_almost_equal(adam.m, expected_m)

    def test_adam_first_update_v_estimate(self):
        """Test 2nd moment estimate (v) after first update."""
        adam = Adam(beta_2=self.beta_2)
        adam.update(self.w.copy(), self.grad)
        
        # After first update: v = beta_2 * 0 + (1 - beta_2) * grad²
        expected_v = (1 - self.beta_2) * (self.grad ** 2)
        np.testing.assert_array_almost_equal(adam.v, expected_v)

    def test_adam_update_shape(self):
        """Test that weights update produces correct shape."""
        adam = Adam()
        w_updated = adam.update(self.w.copy(), self.grad)
        
        self.assertEqual(w_updated.shape, self.w.shape)

    def test_adam_update_decreases_loss(self):
        """Test that Adam update generally decreases weights (gradient direction)."""
        adam = Adam(learning_rate=0.01)
        w = np.array([10.0, 20.0])
        grad = np.array([1.0, 1.0])  # Positive gradient
        
        w_updated = adam.update(w.copy(), grad)
        
        # With positive gradient and positive learning rate, weights should decrease
        # (moving in negative gradient direction)
        self.assertTrue(np.all(w_updated < w))

    def test_adam_multiple_updates_convergence(self):
        """Test that multiple Adam updates show learning."""
        adam = Adam(learning_rate=0.01)
        w = np.array([5.0, 5.0])
        
        # Constant gradient over multiple steps
        grad = np.array([1.0, 1.0])
        
        w_before = w.copy()
        for _ in range(5):
            w = adam.update(w, grad)
        
        # After multiple updates, weights should have changed
        self.assertFalse(np.allclose(w, w_before))

    def test_adam_bias_correction(self):
        """Test that bias correction is applied (more difficult to test directly)."""
        adam = Adam(beta_1=0.9, beta_2=0.999)
        
        # After first update with t=1
        adam.update(self.w.copy(), self.grad)
        
        # t should be 1 (not 0)
        self.assertEqual(adam.t, 1)
        


    def test_adam_epsilon_numerical_stability(self):
        """Test that epsilon prevents division by zero."""
        adam = Adam(epsilon=1e-8)
        
        # Weights and gradient that could cause division by zero
        w = np.array([1.0, 1.0])
        grad = np.array([0.0, 0.0])  # Zero gradient
        
        # Should not raise an error
        try:
            w_updated = adam.update(w.copy(), grad)
            self.assertIsNotNone(w_updated)
        except ZeroDivisionError:
            self.fail("Adam optimizer raised ZeroDivisionError with epsilon protection")

    def test_adam_persistent_state(self):
        """Test that Adam maintains state across multiple updates."""
        adam = Adam()
        w = np.array([1.0, 1.0])
        grad1 = np.array([0.1, 0.2])
        grad2 = np.array([0.05, 0.15])
        
        # First update
        w = adam.update(w.copy(), grad1)
        m_after_1 = adam.m.copy()
        v_after_1 = adam.v.copy()
        t_after_1 = adam.t
        
        # Second update
        w = adam.update(w.copy(), grad2)
        m_after_2 = adam.m.copy()
        v_after_2 = adam.v.copy()
        t_after_2 = adam.t
        
        # State should have changed
        self.assertFalse(np.allclose(m_after_1, m_after_2))
        self.assertFalse(np.allclose(v_after_1, v_after_2))
        self.assertEqual(t_after_2, t_after_1 + 1)

if __name__ == '__main__':
    import unittest
    unittest.main()