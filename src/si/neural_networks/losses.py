from abc import abstractmethod

import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + (1 - y_true) / (1 - p)


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical cross-entropy loss function.

    The categorical cross-entropy loss function in neural networks is applied to
    multi-class classification problems. It measures the dissimilarity between
    predicted class probabilities and true one-hot encoded class labels.

    Exercise 14 (Slide 5)
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss function.

        For multi-class classification with one-hot encoded labels and softmax
        predictions, this computes the negative sum of the true label multiplied
        by the log of the predicted probability.

        Formula:
        L = -∑ y_i * log(y_i*)
        Parameters
        ----------
        y_true : numpy.ndarray
            True labels in one-hot encoded format.
            Shape: (batch_size, n_classes) or (n_classes,)

        y_pred : numpy.ndarray
            Predicted probabilities from softmax activation.
            Shape: (batch_size, n_classes) or (n_classes,)

        Returns
        -------
        float
            The categorical cross-entropy loss value.
        """
        # Avoid log(0) by clipping predictions
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Compute categorical cross-entropy: -sum(y_true * log(y_pred))
        return -np.sum(y_true * np.log(p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the categorical cross-entropy loss function.

        When categorical cross-entropy is combined with softmax activation,
        the derivative simplifies elegantly to:
            dL/d(logits) = - (y_pred / y_true)

        This is because the gradient of softmax combined with cross-entropy
        loss produces this simple form, avoiding the need to compute the
        full Jacobian of softmax.

        Parameters
        ----------
        y_true : numpy.ndarray
            True labels in one-hot encoded format.
            Shape: (batch_size, n_classes) or (n_classes,)

        y_pred : numpy.ndarray
            Predicted probabilities from softmax activation.
            Shape: (batch_size, n_classes) or (n_classes,)

        Returns
        -------
        numpy.ndarray
            The derivative of the loss with respect to the logits.
            Shape: same as y_pred
        """
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / p)