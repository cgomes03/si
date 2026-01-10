import unittest
import numpy as np
from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.ensemble.stacking_classifier import StackingClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier


class TestStackingClassifier(unittest.TestCase):
    """
    Unit tests for the StackingClassifier class
    """

    def setUp(self):
        """
        Set up the test case by loading the dataset and splitting it.
        """
        # Step 1: Use the breast-bin.csv dataset
        self.dataset = read_csv('datasets/breast_bin/breast-bin.csv', features=True, label=True)

        # Step 2: Split the data into train and test sets
        from si.model_selection.split import train_test_split
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_stacking_classifier_fit_predict(self):
        """
        Test the StackingClassifier fit and predict methods.
        """
        # Step 3: Create a KNNClassifier model
        knn_model = KNNClassifier(k=3)

        # Step 4: Create a LogisticRegression model
        logistic_model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000, patience=10, scale=True)

        # Step 5: Create a DecisionTree model
        decision_tree_model = DecisionTreeClassifier(max_depth=10, min_samples_split=2, mode='gini')

        # Step 6: Create a second KNNClassifier model (final model)
        final_knn_model = KNNClassifier(k=5)

        # Step 7: Create a StackingClassifier model using the previous classifiers
        # The second KNNClassifier model must be used as the final model
        stacking_model = StackingClassifier(
            models=[knn_model, logistic_model, decision_tree_model],
            final_model=final_knn_model
        )

        # Step 8: Train the StackingClassifier model
        stacking_model.fit(self.train_dataset)

        # Get predictions
        predictions = stacking_model.predict(self.test_dataset)

        # Check that predictions have the correct shape
        self.assertEqual(predictions.shape[0], self.test_dataset.y.shape[0])

        # Get the score on the test set
        score = stacking_model.score(self.test_dataset)

        # Print the score (as asked in the exercise)
        print(f"\nStackingClassifier score on test set: {score:.4f}")

        # Check that the score is reasonable (between 0 and 1)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Check that the score is at least better than random guessing
        # For binary classification, random guessing would give ~0.5
        self.assertGreater(score, 0.5)

    def test_stacking_classifier_parameters(self):
        """
        Test that the StackingClassifier stores parameters correctly.
        """
        # Create models
        knn_model = KNNClassifier(k=3)
        logistic_model = LogisticRegression()
        final_model = KNNClassifier(k=5)

        # Create StackingClassifier
        stacking_model = StackingClassifier(
            models=[knn_model, logistic_model],
            final_model=final_model
        )

        # Check parameters
        self.assertEqual(len(stacking_model.models), 2)
        self.assertIsInstance(stacking_model.final_model, KNNClassifier)
        self.assertEqual(stacking_model.final_model.k, 5)

    def test_stacking_classifier_multiple_models(self):
        """
        Test StackingClassifier with different numbers of base models.
        """
        # Test with 2 models
        stacking_2_models = StackingClassifier(
            models=[KNNClassifier(k=3), LogisticRegression()],
            final_model=KNNClassifier(k=5)
        )
        stacking_2_models.fit(self.train_dataset)
        score_2 = stacking_2_models.score(self.test_dataset)

        print(f"\nStackingClassifier with 2 base models - score: {score_2:.4f}")

        self.assertGreaterEqual(score_2, 0.0)
        self.assertLessEqual(score_2, 1.0)

        # Test with 3 models (as in the main test)
        stacking_3_models = StackingClassifier(
            models=[
                KNNClassifier(k=3),
                LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000),
                DecisionTreeClassifier(max_depth=10, min_samples_split=2, mode='gini')
            ],
            final_model=KNNClassifier(k=5)
        )
        stacking_3_models.fit(self.train_dataset)
        score_3 = stacking_3_models.score(self.test_dataset)

        print(f"StackingClassifier with 3 base models - score: {score_3:.4f}")

        self.assertGreaterEqual(score_3, 0.0)
        self.assertLessEqual(score_3, 1.0)


if __name__ == '__main__':
    unittest.main()