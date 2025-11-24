from typing import Tuple
import numpy as np
from si.data.dataset import Dataset

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets keeping the proportion of each class.

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
    # Definir a seed para reprodutibilidade (se fornecida)
    if random_state is not None:
        np.random.seed(random_state)

    # 1. Obter as classes únicas e as contagens
    labels, counts = np.unique(dataset.y, return_counts=True)
    
    # Listas para guardar os índices finais
    train_idxs = []
    test_idxs = []

    # 2. Iterar por cada classe para manter a proporção
    for label in labels:
        # Obter todos os índices correspondentes a esta classe
        # np.where devolve um tuple, por isso usamos [0]
        idxs = np.where(dataset.y == label)[0]
        
        # Baralhar os índices desta classe específica
        np.random.shuffle(idxs)
        
        # Calcular quantos elementos desta classe vão para teste
        n_test = int(len(idxs) * test_size)
        
        # Separar os índices
        test_idxs.extend(idxs[:n_test])
        train_idxs.extend(idxs[n_test:])

    # 3. Construir os novos datasets usando os índices recolhidos
    train_dataset = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], 
                            features=dataset.features, label=dataset.label)
    test_dataset = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], 
                           features=dataset.features, label=dataset.label)
    
    return train_dataset, test_dataset
    
