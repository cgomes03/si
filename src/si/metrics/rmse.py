import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between actual and predicted values.
    Formula: sqrt(sum((y_true - y_pred)^2) / N)

    Parameters
    ----------
    y_true: np.ndarray
        The real values.
    y_pred: np.ndarray
        The predicted values.

    Returns
    -------
    float
        The RMSE value.
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))