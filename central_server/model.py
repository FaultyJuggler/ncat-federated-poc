import numpy as np
from sklearn.ensemble import RandomForestClassifier


def create_global_model(params=None):
    """
    Create a new RandomForest model with provided parameters or defaults

    Args:
        params: Dictionary of parameters for RandomForestClassifier

    Returns:
        RandomForestClassifier: Initialized model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'random_state': 42
        }

    return RandomForestClassifier(**params)


def serialize_model(model):
    """Serialize RandomForest model to a dictionary"""
    serialized = {}

    # Extract and serialize tree structures
    trees = []
    for tree in model.estimators_:
        tree_dict = {
            'node_count': tree.tree_.node_count,
            'children_left': tree.tree_.children_left.tolist(),
            'children_right': tree.tree_.children_right.tolist(),
            'feature': tree.tree_.feature.tolist(),
            'threshold': tree.tree_.threshold.tolist(),
            'values': tree.tree_.value.tolist()
        }
        trees.append(tree_dict)

    serialized['trees'] = trees
    serialized['n_classes'] = model.n_classes_
    serialized['n_features'] = model.n_features_in_
    serialized['classes'] = model.classes_.tolist()
    serialized['params'] = {
        'n_estimators': model.n_estimators,
        'criterion': model.criterion,
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'min_samples_leaf': model.min_samples_leaf,
        'bootstrap': model.bootstrap
    }

    return serialized


def merge_forest_weights(models, sample_counts):
    """
    Merge RandomForest models using weighted voting based on sample counts.
    Since we can't directly average RandomForest parameters, we'll merge the trees.
    """
    # Basic validation
    if not models:
        raise ValueError("No models provided for merging")

    # Create a new global model
    global_model = create_global_model({
        'n_estimators': models[0].n_estimators,
        'criterion': models[0].criterion,
        'max_depth': models[0].max_depth,
        'min_samples_split': models[0].min_samples_split,
        'min_samples_leaf': models[0].min_samples_leaf,
        'bootstrap': models[0].bootstrap,
        'random_state': 42
    })

    # Calculate the total number of trees and weights per client
    total_samples = sum(sample_counts)
    weights = [count / total_samples for count in sample_counts]

    # Calculate how many trees to take from each client model
    total_trees = global_model.n_estimators
    trees_per_model = [int(round(weight * total_trees)) for weight in weights]

    # Adjust to ensure we get exactly total_trees
    while sum(trees_per_model) < total_trees:
        idx = trees_per_model.index(min(trees_per_model))
        trees_per_model[idx] += 1
    while sum(trees_per_model) > total_trees:
        idx = trees_per_model.index(max(trees_per_model))
        trees_per_model[idx] -= 1

    # Gather trees from each model according to their weights
    merged_estimators = []
    for model, n_trees in zip(models, trees_per_model):
        if n_trees <= 0:
            continue
        # For simplicity, take the first n_trees
        merged_estimators.extend(model.estimators_[:n_trees])

    # Update the global model's estimators
    global_model.estimators_ = merged_estimators

    # Also need to ensure other attributes are correctly set
    if models:
        reference_model = models[0]
        global_model.n_classes_ = reference_model.n_classes_
        global_model.classes_ = reference_model.classes_
        global_model.n_features_in_ = reference_model.n_features_in_

    return global_model