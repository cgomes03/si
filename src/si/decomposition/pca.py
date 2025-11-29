import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class PCA(Transformer):
    """
    Principal Component Analysis (PCA) using eigenvalue decomposition of the
    covariance matrix.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.

    Estimated attributes
    --------------------
    mean : np.ndarray
        Mean value of each feature in the training data.
    components : np.ndarray
        Principal components; each row is an eigenvector corresponding
        to a principal component.
    explained_variance : np.ndarray
        Ratio of variance explained by each selected principal component.

    Methods
    -------
    _fit(dataset)
        Estimates the mean, principal components and explained variance
        from the dataset.
    _transform(dataset)
        Projects the dataset onto the previously inferred principal
        components, returning a reduced Dataset.
    """

    def __init__(self, n_components: int, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset):
        # 1. Center the data 
        self.mean = np.mean(dataset.X, axis=0)
        X_centered = dataset.X - self.mean

        # 2. Covariance matrix & eigendecomposition 
        covariance_matrix = np.cov(X_centered, rowvar=False)
        # Slides explicitly request np.linalg.eig on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # 3. Sort eigenvalues/eigenvectors (largest variance first) 
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # 4. Infer principal components 
        self.components = sorted_eigenvectors[:, :self.n_components].T

        # 5. Infer explained variance 
        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance = sorted_eigenvalues[:self.n_components] / total_variance

        return self

    def _transform(self, dataset):
        # 1. Center the data using the mean inferred in fit 
        X_centered = dataset.X - self.mean

        # 2. Dimensionality reduction 
        X_reduced = np.dot(X_centered, self.components.T)

        return Dataset(
            X_reduced,
            dataset.y,
            features=[f'PC{i+1}' for i in range(self.n_components)],
            label=dataset.label
        )

