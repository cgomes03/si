import unittest
import numpy as np
from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression


class TestRandomizedSearchCV(unittest.TestCase):
    """
    Unit tests for the randomized_search_cv function.
    """

    def setUp(self):
        """
        Set up the test case by loading the dataset.
        """
        #1: Use the breast-bin.csv dataset
        self.dataset = read_csv('datasets/breast_bin/breast-bin.csv', features=True, label=True)

    def test_randomized_search_cv_protocol(self):
        """
        Test randomized_search_cv following the EXACT protocol from slide 8.
        """
        #2: Create a LogisticRegression model
        model = LogisticRegression()

        #3: Perform a randomized search with the following hyperparameter distributions
        hyperparameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),      # distribution between 1 and 10 with 10 equal intervals
            'alpha': np.linspace(0.001, 0.0001, 100),  # distribution between 0.001 and 0.0001 with 100 equal intervals
            'max_iter': np.linspace(1000, 2000, 200)   # distribution between 1000 and 2000 with 200 equal intervals
        }

        #4: Use n_iter=10 and cv=3 folds for the cross validation
        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=hyperparameter_grid,
            scoring=None,
            cv=3,
            n_iter=10
        )

        #5: Which scores do you obtain? What are the best score and best hyperparameters?
        print("\n" + "=" * 80)
        print("RANDOMIZED SEARCH CV RESULTS (Exercise 8.2)")
        print("=" * 80)

        print(f"\nNumber of combinations tested: {len(results['scores'])}")

        print("\nAll scores obtained:")
        for i, (hyperparams, score) in enumerate(zip(results['hyperparameters'], results['scores']), 1):
            print(f"  {i}. Score: {score:.4f}")
            print(f"     l2_penalty: {hyperparams['l2_penalty']:.4f}")
            print(f"     alpha: {hyperparams['alpha']:.6f}")
            print(f"     max_iter: {hyperparams['max_iter']:.0f}")

        print(f"\nBest score: {results['best_score']:.4f}")
        print(f"Best hyperparameters:")
        print(f"  l2_penalty: {results['best_hyperparameters']['l2_penalty']:.4f}")
        print(f"  alpha: {results['best_hyperparameters']['alpha']:.6f}")
        print(f"  max_iter: {results['best_hyperparameters']['max_iter']:.0f}")

        print("=" * 80)

        # Verify results structure
        self.assertIn('hyperparameters', results)
        self.assertIn('scores', results)
        self.assertIn('best_hyperparameters', results)
        self.assertIn('best_score', results)

        # Verify we tested n_iter combinations (or less if not enough combinations)
        self.assertLessEqual(len(results['scores']), 10)
        self.assertEqual(len(results['hyperparameters']), len(results['scores']))

        # Verify best_hyperparameters contains all expected keys
        self.assertIn('l2_penalty', results['best_hyperparameters'])
        self.assertIn('alpha', results['best_hyperparameters'])
        self.assertIn('max_iter', results['best_hyperparameters'])

        # Verify best_score is reasonable
        self.assertGreater(results['best_score'], 0.0)
        self.assertLessEqual(results['best_score'], 1.0)

        # Verify best_score is actually the maximum score
        self.assertEqual(results['best_score'], max(results['scores']))

    def test_randomized_search_cv_structure(self):
        """
        Test that randomized_search_cv returns the correct structure.
        """
        model = LogisticRegression()

        hyperparameter_grid = {
            'l2_penalty': np.linspace(1, 10, 5),
            'alpha': np.linspace(0.001, 0.0001, 5)
        }

        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=hyperparameter_grid,
            cv=3,
            n_iter=5
        )

        # Check structure
        self.assertIsInstance(results, dict)
        self.assertIsInstance(results['hyperparameters'], list)
        self.assertIsInstance(results['scores'], list)
        self.assertIsInstance(results['best_hyperparameters'], dict)
        self.assertIsInstance(results['best_score'], (int, float, np.number))

    def test_randomized_search_cv_invalid_hyperparameter(self):
        """
        Test that randomized_search_cv raises error for invalid hyperparameters.
        """
        model = LogisticRegression()

        # Invalid hyperparameter that doesn't exist in the model
        hyperparameter_grid = {
            'invalid_param': [1, 2, 3]
        }

        with self.assertRaises(ValueError) as context:
            randomized_search_cv(
                model=model,
                dataset=self.dataset,
                hyperparameter_grid=hyperparameter_grid,
                cv=3,
                n_iter=2
            )

        self.assertIn('does not exist', str(context.exception))

    def test_randomized_search_cv_n_iter_limit(self):
        """
        Test that n_iter correctly limits the number of combinations tested.
        """
        model = LogisticRegression()

        # Create a grid with many combinations
        hyperparameter_grid = {
            'l2_penalty': np.linspace(1, 10, 20),
            'alpha': np.linspace(0.001, 0.0001, 20)
        }
        # Total combinations: 20 * 20 = 400

        # Test with n_iter=5
        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=hyperparameter_grid,
            cv=3,
            n_iter=5
        )

        # Should test exactly 5 combinations
        self.assertEqual(len(results['scores']), 5)
        self.assertEqual(len(results['hyperparameters']), 5)

    def test_randomized_search_cv_small_grid(self):
        """
        Test randomized_search_cv when grid has fewer combinations than n_iter.
        """
        model = LogisticRegression()

        # Small grid with only 4 combinations (2 * 2)
        hyperparameter_grid = {
            'l2_penalty': [1.0, 2.0],
            'alpha': [0.001, 0.01]
        }

        # Request more iterations than available combinations
        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=hyperparameter_grid,
            cv=3,
            n_iter=10
        )

        # Should test all 4 combinations (not 10)
        self.assertEqual(len(results['scores']), 4)


if __name__ == '__main__':
    unittest.main()