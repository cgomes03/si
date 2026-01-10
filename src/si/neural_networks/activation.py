from abc import abstractmethod
from typing import Union

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0
    
class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input >= 0, 1, 0)



class TanhActivation(ActivationLayer):
    """
    Tanh (hyperbolic tangent) activation function.
    
    Squashes values to the range [-1, 1].
    Often preferred over sigmoid in hidden layers.
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Tanh activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer (values in range [-1, 1]).
        """
        return np.tanh(input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivative of the tanh activation function.
        
        Formula: d/dx tanh(x) = 1 - tanh²(x)

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return 1 - np.tanh(input) ** 2


class SoftmaxActivation(ActivationLayer):
    """
    Softmax activation function.

    Transforms arbitrary scores (logits) into a probability distribution
    over classes, where each row sums to 1. Typically used in the output
    layer for multi-class classification problems.
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Compute the numerically stable softmax of the input.

        Implements the STABLE version of softmax by subtracting the maximum
        of each row before applying the exponential, to prevent numerical
        overflow.

        Formula (per sample, i.e., per row x):
            shifted_x = x - max(x)
            softmax_i(x) = exp(shifted_x_i) / sum_j exp(shifted_x_j)

        Why subtract max?
            - exp(x_i) / sum(exp(x_j)) = exp(x_i - c) / sum(exp(x_j - c))
            - Subtracting max(x) prevents overflow of exp() for large values
            - Numerically equivalent but avoids underflow/overflow issues

        Parameters
        ----------
        input : numpy.ndarray
            Input array to the layer.
            Typical shape: (batch_size, n_classes)

        Returns
        -------
        numpy.ndarray
            Array with probability distribution over classes.
            Same shape as input.
            For each row:
                - All values are in [0, 1]
                - Sum of values equals 1
        """
        # Subtract maximum in each row for numerical stability
        x_shifted = input - np.max(input, axis=1, keepdims=True)
        # Exponential of shifted values
        exp_x = np.exp(x_shifted)
        # Normalize to obtain probability distribution
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the softmax activation.

        In theory, the derivative of softmax is a full Jacobian matrix:
            J_ij = softmax_i(x) * (δ_ij - softmax_j(x))

        where δ_ij is the Kronecker delta (1 if i==j, 0 otherwise).

        For practical use in neural networks, especially when combined
        with loss functions like cross-entropy, it is often sufficient
        (or more convenient) to use only the diagonal part of this matrix:

            d_softmax_i / d_x_i ≈ softmax_i(x) * (1 - softmax_i(x))

        This implementation returns this simplified version, element-wise,
        suitable for element-wise multiplication in backward_propagation.

        Parameters
        ----------
        input : numpy.ndarray
            Input array to the layer (same shape as forward).

        Returns
        -------
        numpy.ndarray
            Simplified derivative of softmax with respect to input,
            with the same shape as `input`.
        """
        softmax = self.activation_function(input)
        return softmax * (1 - softmax)
