from typing import Callable
import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile(Transformer):

    """
    Select features based on highest scores given by a scoring function.


    This transformer selects a percentage of top features, based on a
    univariate scoring function (default: f_classification), for supervised feature selection.
    
    Parameters
    ----------
    score_func : Callable
        Function that calculates feature scores (default is f_classification).
    percentile : int, optional
        Percentage of features to select (default=50). Range is (1, 100).
    **kwargs : dict
        Additional arguments to pass to the base Transformer.


    Attributes
    ----------
    F : np.ndarray or None
        Feature scores calculated by the score function.
    p : np.ndarray or None
        p-values for each feature.
    """




    def __init__(self, score_func: Callable = f_classification, percentile=50, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
    
    def _fit(self, dataset: Dataset) -> 'SelectPercentile':


        """
        Compute feature scores for the input dataset.


        Parameters
        ----------
        dataset : Dataset
            The input dataset with features and labels.


        Returns
        -------
        self : SelectPercentile
            Fitted transformer with calculated feature scores.
        """

    def __init__(self, score_func: Callable = f_classification, percentile=50, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        num_features = dataset.X.shape[1]
        k = max(1, int(np.ceil(num_features * self.percentile / 100)))
        indices = np.argsort(self.F)[::-1][:k]
        X_new = dataset.X[:, indices]
        features_new = [dataset.features[i] for i in indices]
        return Dataset(X_new, dataset.y, features_new)

