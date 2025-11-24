from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)


    def dropna(self):
        """
        2.1 - Remove samples (rows) that contain at least one NaN value in any independent variable.
        Returns:
            np.ndarray : Data matrix after removing rows with NaNs.
        """
        has_nan = np.isnan(self.X)
        mask = ~np.any(has_nan, axis=1)
        return Dataset(self.X[mask], self.y[mask] if self.y is not None else None, self.features, self.label)



    def fillna(self, strategy="median"):
        """
        Fill NaN values based on the chosen strategy ('median' or 'mean').
        Args:
            strategy (str): Strategy to fill NaN values, must be 'median' or 'mean'.
        Returns:
            np.ndarray : Data matrix with NaNs filled according to strategy.
        Raises:
            ValueError: If strategy is not 'median' or 'mean'.
        """
        filled_X = self.X.copy()
        nan_indices = np.argwhere(np.isnan(self.X))
        for i, j in nan_indices:
            if strategy == "median":
                # Calculate median ignoring NaNs in the column
                filled_X[i, j] = np.median(self.X[:, j][~np.isnan(self.X[:, j])])
            elif strategy == "mean":
                # Calculate mean ignoring NaNs in the column
                filled_X[i, j] = np.mean(self.X[:, j][~np.isnan(self.X[:, j])])
            else:
                raise ValueError("Strategy must be 'median' or 'mean'.")
        return Dataset(filled_X, self.y, self.features, self.label)


    def remove_by_index(self, index):
        """
        2.3 - Remove samples by specific index (or indices).
        Args:
            index (int or list or np.ndarray): Index/indices of rows to remove.
        Returns:
            np.ndarray : Data matrix with specified rows removed.
        """
        mask = np.ones(self.X.shape[0], dtype=bool)
        mask[index] = False
        return Dataset(self.X[mask], self.y[mask] if self.y is not None else None, self.features, self.label)    



if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print("Dataset shape:", dataset.shape())
    print("Has label?", dataset.has_label())
    print("Classes:", dataset.get_classes())
    print("Mean by feature:", dataset.get_mean())
    print("Variance by feature:", dataset.get_variance())
    print("Median by feature:", dataset.get_median())
    print("Minimum by feature:", dataset.get_min())
    print("Maximum by feature:", dataset.get_max())
    print("Summary:\n", dataset.summary())


    # Iris Dataset Examples
    from si.io.csv_file import read_csv
    iris = read_csv("datasets/iris/iris.csv", features=True, label=True)


    print("\n--- Iris Dataset Examples: ---")


    # Check total number of NaN values in the dataset
    print("Total NaN values in the dataset:", np.isnan(iris.X).sum())


    print("#" * 60)


    # 2.1 - Remove samples containing any NaN values
    print("2.1 - Remove samples with NaN values")
    X_no_nan = iris.dropna()
    print("Shape after removing NaN rows:", X_no_nan.shape())


    print("#" * 60)


    # 2.2 - Fill NaN values using feature median and mean
    print("2.2 - Fill NaN values using median and mean for each feature")
    X_filled_median = iris.fillna(strategy='median')
    X_filled_mean = iris.fillna(strategy='mean')
    print("Shape after filling NaN values (median):", X_filled_median.shape())
    print("Shape after filling NaN values (mean):", X_filled_mean.shape())


    print("#" * 60)


    # 2.3 - Remove sample by specific index
    print("2.3 - Remove sample by specific index")
    index_to_remove = 0  # Example: remove the first sample
    X_removed = iris.remove_by_index(index_to_remove)
    print("Shape after removing sample at index", index_to_remove, ":", X_removed.shape())
