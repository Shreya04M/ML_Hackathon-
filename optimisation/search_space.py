"""
PathSnap - Hyperparameter Search Spaces
Member 3 - Bayesian Hyperparameter Optimisation
"""

SUPPORTED_MODELS = (
    "logistic_regression",
    "knn",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
)


def define_search_space(model_name):
    """Return the hyperparameter search space for a model."""

    search_spaces = {
        "logistic_regression": {
            "C": ("float", 0.0001, 100.0, True),
            "class_weight": ("categorical", [None, "balanced"]),
        },

        "knn": {
            "n_neighbors": ("int", 3, 30),
            "weights": ("categorical", ["uniform", "distance"]),
            "p": ("categorical", [1, 2]),
        },

        "decision_tree": {
            "max_depth": ("int", 3, 30),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "criterion": ("categorical", ["gini", "entropy"]),
            "class_weight": ("categorical", [None, "balanced"]),
        },

        "random_forest": {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 5, 40),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
            "max_features": ("categorical", ["sqrt", "log2", None]),
            "class_weight": ("categorical", [None, "balanced"]),
        },

        "hist_gradient_boosting": {
            "learning_rate": ("float", 0.01, 0.3, True),
            "max_iter": ("int", 50, 300),
            "max_leaf_nodes": ("int", 15, 63),
            "min_samples_leaf": ("int", 5, 40),
            "l2_regularization": ("float", 0.000001, 10.0, True),
        },
    }

    if model_name not in search_spaces:
        raise ValueError(f"Unsupported model: {model_name}")

    return search_spaces[model_name]


def suggest_parameters(trial, search_space):
    """Generate hyperparameters using an Optuna trial."""

    parameters = {}

    for name, specification in search_space.items():

        parameter_type = specification[0]

        if parameter_type == "int":
            parameters[name] = trial.suggest_int(
                name, specification[1], specification[2]
            )

        elif parameter_type == "float":
            parameters[name] = trial.suggest_float(
                name,
                specification[1],
                specification[2],
                log=specification[3],
            )

        elif parameter_type == "categorical":
            parameters[name] = trial.suggest_categorical(
                name, specification[1]
            )

        else:
            raise ValueError(
                f"Unsupported parameter type: {parameter_type}"
            )

    return parameters