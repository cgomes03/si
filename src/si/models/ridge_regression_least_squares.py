import numpy as np
from si.model_selection.split import train_test_split
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression using the closed-form Least Squares solution.

    This model fits linear regression parameters with L2 regularization 
    (Ridge penalty), solving the normal equation:

        θ = (XᵀX + λI)⁻¹ Xᵀy

    Scaling can be applied to the features before fitting the model.
    """

    def __init__(self, l2_penalty: float = 1, scale: bool = True, **kwargs):
        """
        Parameters
        ----------
        l2_penalty : float
            Regularization strength (λ). Higher values increase shrinkage.
        scale : bool
            If True, standardizes features before fitting.
        kwargs : dict
            Additional keyword arguments for the parent Model class.
        """
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale

        # Model parameters
        self.theta = None         # weights
        self.theta_zero = None    # intercept term
        self.mean = None          # feature means (for scaling)
        self.std = None           # feature std deviations (for scaling)

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the Ridge Regression model to the dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing features X and target y.

        Returns
        -------
        self : RidgeRegressionLeastSquares
            The fitted model.
        """
        # Scale features if requested
        if self.scale:
            self.mean = np.mean(dataset.X, axis=0)
            self.std = np.std(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # Add intercept term
        X_i = np.c_[np.ones(X.shape[0]), X]

        # Ridge penalty matrix (no penalty for intercept)
        I = np.eye(X_i.shape[1])
        penalty_matrix = self.l2_penalty * I
        penalty_matrix[0, 0] = 0

        # Closed-form Ridge Regression solution
        theta_full = np.linalg.inv(X_i.T @ X_i + penalty_matrix) @ X_i.T @ dataset.y

        # Separate intercept and weights
        self.theta_zero = theta_full[0]
        self.theta = theta_full[1:]
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict target values for a given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing features X.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        # Apply scaling if used during training
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # Add intercept term
        X_i = np.c_[np.ones(X.shape[0]), X]

        # Compute predictions
        return X_i @ np.concatenate(([self.theta_zero], self.theta))

    def _score(self, dataset: Dataset, predictions: np.ndarray = None) -> float:
        """
        Compute the Mean Squared Error (MSE) on a given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing true labels y.
        predictions : np.ndarray, optional
            Precomputed predictions. If None, predictions are computed.

        Returns
        -------
        float
            The MSE score.
        """
        if predictions is None:
            predictions = self._predict(dataset)
        return mse(dataset.y, predictions)


if __name__ == "__main__":
    # Example usage
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([3, 5, 7, 9, 11])

    dataset = Dataset(X, y)

    # train_test_split returns 2 Dataset objects
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )

    model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
    model.fit(train_dataset)

    predictions = model.predict(test_dataset)
    score = model.score(test_dataset)

    print("Predictions:", predictions)
    print("MSE Score:", score)

