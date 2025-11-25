from typing import Callable, Union

import numpy as np

from si.metrics.rmse import rmse
from si.base.model import Model
from si.data.dataset import Dataset


class KNNRegressor(Model):

    def __init__(self, k: int = 1, distance: Callable = rmse, **kwargs):
        """
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self




    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        
        predictions: np.ndarray
            An array with the predictions 

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        predictions = []
        
        for sample in dataset.X:
            distances = self.distance(sample, self.dataset.X)
            
            k_nearest_indices = np.argsort(distances)[:self.k]
            
            k_nearest_values = self.dataset.y[k_nearest_indices]
            
            prediction = np.mean(k_nearest_values)
            predictions.append(prediction)
            
        return np.array(predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the RMSE between the predictions and the actual values.
        """
        return rmse(dataset.y, predictions)










if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.io.csv_file import read_csv


    # load and split the dataset
    dataset = read_csv('datasets/cpu/cpu.csv', features=True, label=True)
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNRegressor(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The rmse of the model is: {score}')