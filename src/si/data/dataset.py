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


    def dropna(self)-> 'Dataset':
        """
        Removes all samples containing at least one null value (NaN).

        Returns
        -------
        Dataset
            The modified dataset without samples containing NaN values.
        """
        # Check for rows with at least one NaN
        has_nan = np.isnan(self.X).any(axis=1)
        
        # Invert mask to get rows to keep
        mask = ~has_nan
        
        # Filter X
        self.X = self.X[mask]
        
        # Filter y if it exists
        if self.y is not None:
            self.y = self.y[mask]
            
        return self


    def fillna(self, value: Union[float, str] = "mean") -> 'Dataset':
        """
        Replaces all null values with another value or the mean or median of the feature/variable.

        Parameters
        ----------
        value: float or str, default="mean"
            The value to replace NaN values with. Can be a specific float value, 'mean', or 'median'.

        Returns
        -------
        Dataset
            The modified dataset with filled NaN values.
        """
        if isinstance(value, str):
            if value == 'mean':
                # Calculate mean for each column ignoring NaNs
                col_means = np.nanmean(self.X, axis=0)
                
                # Find indices where NaN exists
                inds = np.where(np.isnan(self.X))
                
                # Replace NaNs with the corresponding column mean
                self.X[inds] = np.take(col_means, inds[1])
                
            elif value == 'median':
                # Calculate median for each column ignoring NaNs
                col_medians = np.nanmedian(self.X, axis=0)
                
                # Find indices where NaN exists
                inds = np.where(np.isnan(self.X))
                
                # Replace NaNs with the corresponding column median
                self.X[inds] = np.take(col_medians, inds[1])
            else:
                raise ValueError("Value must be a float, 'mean', or 'median'")
        elif isinstance(value, (int, float)):
            # Replace all NaNs with the provided scalar value
            self.X = np.nan_to_num(self.X, nan=value)
        else:
             raise ValueError("Value must be a float, 'mean', or 'median'")
            
        return self

    def remove_by_index(self, index: int) -> 'Dataset':
        """
        Removes a sample by its index.

        Parameters
        ----------
        index: int
            The index of the sample to remove.

        Returns
        -------
        Dataset
            The modified dataset without the removed sample.
        """
        # Remove sample from X
        self.X = np.delete(self.X, index, axis=0)
        
        # Remove sample from y if it exists
        if self.y is not None:
            self.y = np.delete(self.y, index, axis=0)
            
        return self



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




    # --- Iris Dataset Examples ----
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
    X_filled_median = iris.fillna(value='median')
    X_filled_mean = iris.fillna(value='mean')
    print("Shape after filling NaN values (median):", X_filled_median.shape())
    print("Shape after filling NaN values (mean):", X_filled_mean.shape())


    print("#" * 60)


    # 2.3 - Remove sample by specific index
    print("2.3 - Remove sample by specific index")
    index_to_remove = 0  # Example: remove the first sample
    X_removed = iris.remove_by_index(index_to_remove)
    print("Shape after removing sample at index", index_to_remove, ":", X_removed.shape())
