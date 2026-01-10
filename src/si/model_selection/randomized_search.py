import numpy as np
import itertools
from si.io.csv_file import read_csv
from si.model_selection.cross_validate import k_fold_cross_validation

from si.data.dataset import Dataset
from si.models.logistic_regression import LogisticRegression


def randomized_search_cv(model, dataset, hyperparameter_grid, scoring=None, cv=5, n_iter=10):
    """
    Randomized search on hyperparameters with cross-validation.

    The randomized_search_cv function implements a parameter optimization strategy with 
    cross validation using a number of random combinations selected from a distribution 
    of possible hyperparameters.

    Parameters
    ----------
    model : Model
        The model to validate
    dataset : Dataset
        The validation dataset
    hyperparameter_grid : dict
        Dictionary with the hyperparameter name and search values (distributions)
    scoring : function, optional
        Score function. If None, uses the model's score method.
    cv : int, default=5
        Number of folds
    n_iter : int, default=10
        Number of hyperparameter random combinations to test

    Returns
    -------
    dict
        Dictionary with the results of the randomized search cross validation.
        Includes the scores, hyperparameters, best hyperparameters and best score.
        Contains:
        - 'hyperparameters': list of hyperparameters used
        - 'scores': list of scores obtained
        - 'best_hyperparameters': best combination of hyperparameters
        - 'best_score': best score
    """
    #1: Check if the provided hyperparameters are valid (exist in the model)
    for hyperparam in hyperparameter_grid.keys():
        if not hasattr(model, hyperparam):
            raise ValueError(f"Hyperparameter '{hyperparam}' does not exist in the model")

    #2: Get n_iter hyperparameter combinations
    # First, get all possible combinations
    hyperparameter_names = list(hyperparameter_grid.keys())
    hyperparameter_values = list(hyperparameter_grid.values())

    # Generate all possible combinations
    all_combinations = list(itertools.product(*hyperparameter_values))

    # Randomly select n_iter combinations using np.random.choice
    if len(all_combinations) <= n_iter:
        # If we have fewer combinations than n_iter, use all of them
        selected_combinations = all_combinations
    else:
        # Randomly select n_iter combinations without replacement
        selected_indices = np.random.choice(len(all_combinations), size=n_iter, replace=False)
        selected_combinations = [all_combinations[i] for i in selected_indices]

    # Store results
    hyperparameters_list = []
    scores_list = []
    best_score = -np.inf
    best_hyperparameters = None

    # Steps 3-6: For each hyperparameter combination
    for combination in selected_combinations:
        #3: Set the model hyperparameters with the current combination (using setattr)
        current_hyperparameters = {}
        for i, hyperparam_name in enumerate(hyperparameter_names):
            hyperparam_value = combination[i]
            setattr(model, hyperparam_name, hyperparam_value)
            current_hyperparameters[hyperparam_name] = hyperparam_value

        #4: Cross validate the model using k_fold_cross_validation function
        scores = k_fold_cross_validation(model, dataset, scoring=scoring, cv=cv)

        #5: Save the mean of the scores (k scores for k folds) and respective hyperparameters
        mean_score = np.mean(scores)
        hyperparameters_list.append(current_hyperparameters)
        scores_list.append(mean_score)

        #7: Track best score and respective hyperparameters
        if mean_score > best_score:
            best_score = mean_score
            best_hyperparameters = current_hyperparameters.copy()

    #8: Return dictionary including all scores, hyperparameters, best score and best hyperparameters
    return {
        'hyperparameters': hyperparameters_list,
        'scores': scores_list,
        'best_hyperparameters': best_hyperparameters,
        'best_score': best_score
    }


"""
Example usage of randomized_search_cv function.
"""
print("RANDOMIZED SEARCH CV - Example Following Slide 8 Protocol")
print("=" * 80)

#1: Use the breast-bin.csv dataset
dataset = read_csv('datasets/breast_bin/breast-bin.csv', features=True, label=True)
print(f"Dataset shape: {dataset.shape()}")

#2: Create a LogisticRegression model
model = LogisticRegression()

#3: Perform a randomized search with the following hyperparameter distributions
print("\Hyperparameter Distributions:")
hyperparameter_grid = {
    'l2_penalty': np.linspace(1, 10, 10),      # distribution between 1 and 10 with 10 equal intervals
    'alpha': np.linspace(0.001, 0.0001, 100),  # distribution between 0.001 and 0.0001 with 100 equal intervals
    'max_iter': np.linspace(1000, 2000, 200)   # distribution between 1000 and 2000 with 200 equal intervals
}

print(f"  l2_penalty: {len(hyperparameter_grid['l2_penalty'])} values from {hyperparameter_grid['l2_penalty'][0]:.1f} to {hyperparameter_grid['l2_penalty'][-1]:.1f}")
print(f"  alpha: {len(hyperparameter_grid['alpha'])} values from {hyperparameter_grid['alpha'][0]:.6f} to {hyperparameter_grid['alpha'][-1]:.6f}")
print(f"  max_iter: {len(hyperparameter_grid['max_iter'])} values from {hyperparameter_grid['max_iter'][0]:.0f} to {hyperparameter_grid['max_iter'][-1]:.0f}")
print(f"  Total possible combinations: {len(hyperparameter_grid['l2_penalty']) * len(hyperparameter_grid['alpha']) * len(hyperparameter_grid['max_iter']):,}")

#4: Use n_iter=10 and cv=3 folds for the cross validation
results = randomized_search_cv(
    model=model,
    dataset=dataset,
    hyperparameter_grid=hyperparameter_grid,
    scoring=None,
    cv=3,
    n_iter=10
)

#5: Which scores do you obtain? What are the best score and best hyperparameters?
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nNumber of combinations tested: {len(results['scores'])}")

print("\nAll scores obtained:")
for i, (hyperparams, score) in enumerate(zip(results['hyperparameters'], results['scores']), 1):
    print(f"\n  Combination {i}:")
    print(f"    Score: {score:.4f}")
    print(f"    l2_penalty: {hyperparams['l2_penalty']:.4f}")
    print(f"    alpha: {hyperparams['alpha']:.6f}")
    print(f"    max_iter: {hyperparams['max_iter']:.0f}")

print("\n" + "=" * 80)
print("BEST RESULTS")
print("=" * 80)
print(f"\nBest score: {results['best_score']:.4f}")
print(f"\nBest hyperparameters:")
print(f"  l2_penalty: {results['best_hyperparameters']['l2_penalty']:.4f}")
print(f"  alpha: {results['best_hyperparameters']['alpha']:.6f}")
print(f"  max_iter: {results['best_hyperparameters']['max_iter']:.0f}")


# Efficiency
total_combinations = len(hyperparameter_grid['l2_penalty']) * len(hyperparameter_grid['alpha']) * len(hyperparameter_grid['max_iter'])
tested_combinations = len(results['scores'])
print(f"\nSearch efficiency:")
print(f"  Tested: {tested_combinations} out of {total_combinations:,} possible combinations")
print(f"  Coverage: {tested_combinations / total_combinations * 100:.4f}%")
