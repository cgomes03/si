import os
import unittest
import numpy as np

from evaluation.exercises.Exercise6 import stratified_train_test_split
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv

DATASETS_PATH = "datasets"  

class TestStratifiedTrainTestSplit(unittest.TestCase):

    def setUp(self):
        # Caminho para o dataset Iris
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_stratified_split(self):
        # Definir seed localmente para garantir reprodutibilidade neste teste
        np.random.seed(42)
        
        # Executar a divisão (30% para teste)
        train_dataset, test_dataset = stratified_train_test_split(self.dataset, test_size=0.3, random_state=42)
        
        # 1. Validação de tamanhos (Shapes)
        # Iris tem 150 amostras. 30% de 150 = 45 para teste, 105 para treino.
        self.assertEqual(train_dataset.shape()[0], 105)
        self.assertEqual(test_dataset.shape()[0], 45)
        
        # 2. Validação de proporções (Estratificação)
        # Proporções originais
        unique, counts = np.unique(self.dataset.y, return_counts=True)
        original_proportions = counts / len(self.dataset.y)
        
        # Proporções no Treino
        train_unique, train_counts = np.unique(train_dataset.y, return_counts=True)
        train_proportions = train_counts / len(train_dataset.y)
        
        # Proporções no Teste
        test_unique, test_counts = np.unique(test_dataset.y, return_counts=True)
        test_proportions = test_counts / len(test_dataset.y)   
        
        # Verificar se as proporções se mantêm (com margem de erro de 0.05)
        for i in range(len(unique)):
            # Nota: Usamos assertTrue com np.isclose para comparar floats
            self.assertTrue(np.isclose(original_proportions[i], train_proportions[i], atol=0.05))
            self.assertTrue(np.isclose(original_proportions[i], test_proportions[i], atol=0.05))

if __name__ == '__main__':
    unittest.main()