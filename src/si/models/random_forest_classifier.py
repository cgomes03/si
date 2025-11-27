from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.base.model import Model
import numpy as np

from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.model_selection.split import train_test_split


class RandomForestClassifier(Model):
    """
    Random forest classifier.

    Ensemble of multiple DecisionTreeClassifier models trained on
    bootstrap samples of the dataset and random subsets of features.
    The final prediction is obtained by majority voting.
    """
    def __init__(self, n_estimators: int = 100, max_features: int = None, min_samples_split: int = 2, max_depth: int = None,
                 mode: str = "entropy", seed: int = 42):
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest.
        max_features : int, optional
            Number of features to consider for each tree.
            If None, uses int(sqrt(n_features)).
        min_samples_split : int
            Minimum number of samples required to split an internal node.
        max_depth : int, optional
            Maximum depth of each tree. If None, trees grow until pure or
            until min_samples_split is reached.
        mode : str
            Impurity criterion for the DecisionTree ("gini" or "entropy").
        seed : int
            Random seed for reproducibility.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # list of (feature_indices, trained_tree)
        self.trees = []

    def _fit(self, dataset: Dataset):
        """
        Fit the random forest on the given dataset.

        For each tree:
        - draw a bootstrap sample of the data (with replacement)
        - choose a random subset of features (without replacement)
        - train a DecisionTreeClassifier on that bootstrapped dataset
        """
        # set seed for reproducibility
        np.random.seed(self.seed)

        # if max_features is None, use sqrt(n_features)
        n_features = dataset.X.shape[1]
        max_features = (
            int(np.sqrt(n_features))
            if self.max_features is None
            else self.max_features
        )

        for _ in range(self.n_estimators):
            # bootstrap samples (with replacement)
            n_samples = dataset.X.shape[0]
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = dataset.X[sample_indices]
            y_bootstrap = dataset.y[sample_indices]

            # random subset of features (without replacement)
            feature_indices = np.random.choice(n_features, size=max_features, replace=False)
            X_bootstrap = X_bootstrap[:, feature_indices]

            # build bootstrap Dataset
            feature_names = [dataset.features[i] for i in feature_indices]
            bootstrap_dataset = Dataset(X_bootstrap,y_bootstrap,features=feature_names,label=dataset.label,)

            # train one decision tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,mode=self.mode,)
            tree.fit(bootstrap_dataset)

            # store (used feature indices, trained tree)
            self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset):
        """
        Predict class labels for the given dataset using majority voting.

        Returns
        -------
        np.ndarray
            Array of predicted labels, one per sample.
        """
        # store each tree's predictions (object allows string labels)
        tree_predictions = np.empty((dataset.X.shape[0], self.n_estimators), dtype=object,)

        # get predictions from each tree
        for i, (feature_indices, tree) in enumerate(self.trees):
            X_subset = dataset.X[:, feature_indices]
            temp_dataset = Dataset(X_subset,
                                   dataset.y,  # labels present but not used in prediction
                                   features=[dataset.features[j] for j in feature_indices], label=dataset.label,)
            predictions = tree.predict(temp_dataset)
            tree_predictions[:, i] = predictions

        # majority vote for each sample
        final_predictions = []
        for i in range(dataset.X.shape[0]):
            values, counts = np.unique(tree_predictions[i, :], return_counts=True)
            final_predictions.append(values[np.argmax(counts)])

        return np.array(final_predictions)

    def _score(self, dataset: Dataset, predictions) -> float:
        """
        Compute the accuracy of the given predictions on the dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the true labels in y.
        predictions : np.ndarray
            Model predictions for this dataset.

        Returns
        -------
        float
            Accuracy = proportion of correct predictions.
        """
        accuracy = np.sum(predictions == dataset.y) / len(dataset.y)
        return accuracy


# Example usage
if __name__ == "__main__":
    # Load dataset
    dataset = read_csv("datasets/iris/iris.csv", features=True, label=True)

    # Split dataset
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create and train RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=10,max_features=2,max_depth=5,min_samples_split=2,seed=42,)
    rfc.fit(train_dataset)

    # Make predictions
    predictions = rfc.predict(test_dataset)

    # Evaluate accuracy
    accuracy = rfc.score(test_dataset)
    print(f"Accuracy: {accuracy}")
