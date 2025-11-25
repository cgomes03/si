from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets keeping the class proportion.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # 1. Get unique class labels and their counts 
    unique_labels, labels_counts = np.unique(dataset.y, return_counts=True)
    
    # 2. Initialize empty lists for train and test indices 
    train_idxs = []
    test_idxs = []
    
    # Set random state for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # 3. Loop through unique labels 
    for i, label in enumerate(unique_labels):
        # Get indices for the current class
        label_idxs = np.where(dataset.y == label)[0]
        
        # 4. Calculate the number of test samples for the current class 
        n_test = int(labels_counts[i] * test_size)
        
        # 5. Shuffle and select indices for the current class 
        np.random.shuffle(label_idxs)
        
        # Add to test indices
        test_idxs.extend(label_idxs[:n_test])
        
        # 6. Add the remaining indices to the train indices 
        train_idxs.extend(label_idxs[n_test:])
    
    # 7. Create training and testing datasets 
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    
    # 8. Return the training and testing datasets 
    return train, test