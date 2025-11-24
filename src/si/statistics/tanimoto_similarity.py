import numpy as np
import sklearn


def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    """
    Compute the Tanimoto similarity between a single vector x and each row vector in a matrix y.

    The Tanimoto similarity (also known as Jaccard similarity for binary vectors) between two vectors is defined as:
        S(x, y) = (x . y) / [sum(x^2) + sum(y^2) - sum(x * y)]
    where '.' denotes the dot product.

    Parameters
    ----------
    x : np.ndarray
        A 1-dimensional array representing the reference vector.
    y : np.ndarray
        A 2-dimensional array where each row is a vector to compare with x.

    Returns
    -------
    np.ndarray
        A 1-dimensional array of Tanimoto similarity values between x and each row of y.
    """

    return np.dot(y, x) / (x**2 + y**2 - x*y).sum(axis=1)