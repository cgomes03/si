from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    



class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    """

    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, 
                 beta_2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Adam optimizer.

        Parameters
        ----------
        learning_rate : float
            The learning rate to use for updating the weights. Default: 0.001
        
        beta_1 : float
            The exponential decay rate for the 1st moment estimates (moving average
            of gradients). Controls how much past gradients influence the current update.
            Default: 0.9
        
        beta_2 : float
            The exponential decay rate for the 2nd moment estimates (moving average
            of squared gradients). Controls the adaptive learning rate scaling.
            Default: 0.999
        
        epsilon : float
            A small constant for numerical stability to prevent division by zero.
            Default: 1e-8
        """
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        # Estimated parameters (initialized on first call to update)
        self.m = None  # 1st moment estimate (moving average of gradients)
        self.v = None  # 2nd moment estimate (moving average of squared gradients)
        self.t = 0     # Time step (epoch counter), initialized to 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights using the Adam algorithm.

        Parameters
        ----------
        w : numpy.ndarray
            The current weights of the layer.
        
        grad_loss_w : numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        #1: Initialize m and v if not already done
        if self.m is None:
            self.m = np.zeros(np.shape(w))
        if self.v is None:
            self.v = np.zeros(np.shape(w))
        
        #2: Update time step
        self.t += 1
        
        #3: Update biased 1st moment estimate (exponential moving average of gradients)
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        
        #4: Update biased 2nd moment estimate (exponential moving average of squared gradients)
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)
        
        #5: Compute bias-corrected 1st moment estimate
        # Early in training, m is biased towards zero, so we correct this
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        
        #6: Compute bias-corrected 2nd moment estimate
        # Early in training, v is biased towards zero, so we correct this
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        
        #7: Update weights using the corrected moment estimates
        # The adaptive learning rate is scaled by 1 / (sqrt(v_hat) + epsilon)
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)