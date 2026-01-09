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
        """
        Initialize the SelectPercentile transformer.
        """

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

        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects a given percentage of features based on their F-values.

        This method uses the feature scores stored in ``self.F`` to select
        the top features according to the specified percentile. It computes
        a threshold using the corresponding percentile of the F-values and:
        - first selects all features with F strictly greater than the threshold;
        - if needed, adds features tied at the threshold (F equal to the threshold)
        until the exact number of features defined by the percentile is reached.

        Parameters
        ----------
        dataset : Dataset
            The input dataset to select features from.

        Returns
        -------
        Dataset
            A new Dataset object containing only the selected features.
        """

        # total number of features
        num_features = dataset.X.shape[1]

        # number of features to select 
        num_features_to_select = int(np.ceil(num_features * self.percentile / 100))

        # percentile corresponding to the threshold 
        threshold_percentile = 100 - self.percentile
        threshold = np.percentile(self.F, threshold_percentile)

        # indices of features with F strictly greater than the threshold
        mask_strict = self.F > threshold
        strict_indices = np.where(mask_strict)[0]

        # if enough features, keep only the top ones by F
        if strict_indices.size >= num_features_to_select:
            # sort by F within the strict indices and take the top k
            order = np.argsort(self.F[strict_indices])[-num_features_to_select:]
            selected_indices = strict_indices[order]
        else:
            # still missing features: include those tied at the threshold 
            mask_ties = self.F == threshold
            tie_indices = np.where(mask_ties)[0]

            remaining = num_features_to_select - strict_indices.size

            selected_ties = tie_indices[:remaining]

            selected_indices = np.concatenate([strict_indices, selected_ties])

        selected_indices = np.sort(selected_indices)

        features_new = [dataset.features[i] for i in selected_indices]
        X_new = dataset.X[:, selected_indices]

        return Dataset(X_new, dataset.y, features_new, dataset.label)


