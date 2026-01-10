import numpy as np
from si.data.dataset import Dataset
from si.base.model import Model
from si.metrics.accuracy import accuracy

from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.model_selection.split import train_test_split
from si.io.csv_file import read_csv




class StackingClassifier(Model):
    """
    StackingClassifier ensemble model.
    The StackingClassifier harnesses an ensemble of models to generate predictions. 
    These predictions are subsequently employed to train another model – the final model. 
    The final model can then be used to predict the output variable (Y).

    Parameters
    ----------
    models : list
        Initial set of models
    final_model : Model
        The model to make the final predictions
    """

    def __init__(self, models, final_model):
        """
        Initialize the StackingClassifier.

        Parameters
        ----------
        models : list
            Initial set of models
        final_model : Model
            The model to make the final predictions
        """
        super().__init__()
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset):
        """
        Train the ensemble models.

        Algorithm:
        1. Train the initial set of models
        2. Get predictions from the initial set of models
        3. Train the final model with the predictions of the initial set of models
        4. Returns itself (self)

        Parameters
        ----------
        dataset : Dataset
            The dataset to train the models

        Returns
        -------
        self : StackingClassifier
            The fitted model
        """
        #1: Train the initial set of models
        for model in self.models:
            model.fit(dataset)

        #2: Get predictions from the initial set of models
        predictions = []
        for model in self.models:
            pred = model.predict(dataset)
            predictions.append(pred)

        # Stack predictions as columns (each model's predictions as a column)
        stacked_predictions = np.column_stack(predictions)

        #3: Train the final model with the predictions of the initial set of models
        # Create a new dataset with the stacked predictions as X and original y
        from si.data.dataset import Dataset
        meta_dataset = Dataset(X=stacked_predictions, y=dataset.y)
        self.final_model.fit(meta_dataset)

        #4: Return itself
        return self

    def _predict(self, dataset):
        """
        Predicts the labels using the ensemble models.

        Algorithm:
        1. Get predictions from the initial set of models
        2. Get the final predictions using the final model and the predictions 
           of the initial set of models

        Parameters
        ----------
        dataset : np.ndarray
            The dataset to make predictions on

        Returns
        -------
        predictions : np.ndarray
            The predicted labels
        """
        #1: Get predictions from the initial set of models
        predictions = []
        for model in self.models:
            pred = model.predict(dataset)
            predictions.append(pred)

        # Stack predictions as columns
        stacked_predictions = np.column_stack(predictions)

        #2: Get the final predictions using the final model
        meta_dataset = Dataset(X=stacked_predictions, y=dataset.y)
        final_predictions = self.final_model.predict(meta_dataset)

        return final_predictions

    def _score(self, dataset, predictions):
        """
        Computes the accuracy between predicted and real labels.

        Algorithm:
        1. Get predictions using the predict method
        2. Computes the accuracy between predicted and real values

        Parameters
        ----------
        dataset : Dataset
            The dataset to score
        predictions : np.ndarray
            The predicted labels

        Returns
        -------
        accuracy : float
            The accuracy score
        """
        #1: Get predictions using the predict method
        #2: Compute the accuracy between predicted and real values
        return accuracy(dataset.y, predictions)
    



"""
Example usage of StackingClassifier following the testing protocol.
"""

#1: Use the breast-bin.csv dataset
dataset = read_csv('datasets/breast_bin/breast-bin.csv', features=True, label=True)

#2: Split the data into train and test sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

#3: Create a KNNClassifier model
knn_model = KNNClassifier(k=3)

#4: Create a LogisticRegression model
logistic_model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000, patience=10, scale=True)

#5: Create a DecisionTree model
decision_tree_model = DecisionTreeClassifier(max_depth=10, min_samples_split=2, mode='gini')

#6: Create a second KNNClassifier model (final model)
final_knn_model = KNNClassifier(k=5)

#7: Create a StackingClassifier model using the previous classifiers
# The second KNNClassifier model must be used as the final model
stacking_model = StackingClassifier(
    models=[knn_model, logistic_model, decision_tree_model],
    final_model=final_knn_model
)

#8: Train the StackingClassifier model
print("Training StackingClassifier...")
stacking_model.fit(train_dataset)


# What is the score of the model on the test set?
test_score = stacking_model.score(test_dataset)
print(f"StackingClassifier score on test set: {test_score:.4f}")

# Also compute scores for individual models for comparison
print("\nComparison with individual models:")

# KNN
knn_comparison = KNNClassifier(k=3)
knn_comparison.fit(train_dataset)
knn_score = knn_comparison.score(test_dataset)
print(f"KNN (k=3) score: {knn_score:.4f}")

# Logistic Regression
lr_comparison = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000, patience=10, scale=True)
lr_comparison.fit(train_dataset)
lr_score = lr_comparison.score(test_dataset)
print(f"Logistic Regression score: {lr_score:.4f}")

# Decision Tree
dt_comparison = DecisionTreeClassifier(max_depth=10, min_samples_split=2, mode='gini')
dt_comparison.fit(train_dataset)
dt_score = dt_comparison.score(test_dataset)
print(f"Decision Tree score: {dt_score:.4f}")

print(f"\nStacking Ensemble score: {test_score:.4f}")
print(f"Improvement over best individual model: {test_score - max(knn_score, lr_score, dt_score):.4f}")