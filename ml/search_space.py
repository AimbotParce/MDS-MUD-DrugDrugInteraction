import yaml
from ezautoml.evaluation.task import TaskType
from ezautoml.space.component import Component, Tag
from ezautoml.space.search_space import SearchSpace
from ezautoml.space.hyperparam import Hyperparam 
from ezautoml.space.space import Integer, Real, Categorical
from ezautoml.registry import constructor_registry

# -----------------------------
# Exclude Naive Bayes model
# -----------------------------
classification_model_names = [
    "RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression", 
    "KNeighborsClassifier", "DecisionTreeClassifier",
    "AdaBoostClassifier", "BaggingClassifier", "ExtraTreesClassifier",
    "XGBClassifier", "LGBMClassifier"
]

# -----------------------------
# Define null components
# -----------------------------
null_components = [
    ("no_data_proc", "NoDataProcessing", TaskType.BOTH, Tag.DATA_PROCESSING),
    ("no_feat_proc", "NoFeatureProcessing", TaskType.BOTH, Tag.FEATURE_PROCESSING)
]

# -----------------------------
# Helper functions
# -----------------------------
def get_registered_components(model_names, task):
    components = []
    for name in model_names:
        if not constructor_registry.has(name):
            continue

        constructor = constructor_registry.get(name)

        # Shared hyperparameter templates with generous but valid ranges
        rf_tree_common = [
            Hyperparam("n_estimators", Integer(50, 2000)),  # More trees for stability
            Hyperparam("max_depth", Integer(1, 100)),
            Hyperparam("min_samples_split", Integer(2, 50)),
            Hyperparam("min_samples_leaf", Integer(1, 30))
        ]
        boosting_common = [
            Hyperparam("n_estimators", Integer(50, 2000)),
            Hyperparam("learning_rate", Real(0.001, 1.0)),  # Covers slow to aggressive learning
            Hyperparam("max_depth", Integer(1, 50))
        ]
        bagging_common = [
            Hyperparam("n_estimators", Integer(10, 1000)),
            Hyperparam("max_samples", Real(0.1, 1.0)),
            Hyperparam("max_features", Real(0.1, 1.0))
        ]

        if name == "RandomForestClassifier":
            hyperparams = rf_tree_common + [
                Hyperparam("max_features", Categorical(["sqrt", "log2", None]))
            ]
        elif name == "GradientBoostingClassifier":
            hyperparams = boosting_common + [
                Hyperparam("subsample", Real(0.3, 1.0))  # Wider subsample range
            ]
        elif name == "LogisticRegression":
            hyperparams = [
                Hyperparam("C", Real(1e-2, 1e2)),  # log-scale L2 regularization
                Hyperparam("max_iter", Integer(100, 5000)),
                Hyperparam("penalty", Categorical(["l2"]))
            ]
        elif name == "KNeighborsClassifier":
            hyperparams = [
                Hyperparam("n_neighbors", Integer(1, 50)),  # Wider neighbor choices
                Hyperparam("weights", Categorical(["uniform", "distance"])),
                Hyperparam("leaf_size", Integer(10, 200)),
                Hyperparam("p", Integer(1, 5))  # Allow Minkowski distance generalization
            ]
        elif name == "DecisionTreeClassifier":
            hyperparams = [
                Hyperparam("criterion", Categorical(["gini", "entropy", "log_loss"])),
                Hyperparam("max_depth", Integer(1, 100)),
                Hyperparam("min_samples_split", Integer(2, 50)),
                Hyperparam("min_samples_leaf", Integer(1, 30))
            ]
        elif name == "AdaBoostClassifier":
            hyperparams = [
                Hyperparam("n_estimators", Integer(50, 2000)),
                Hyperparam("learning_rate", Real(0.001, 2.0))  # Allow very slow learning
            ]
        elif name == "BaggingClassifier":
            hyperparams = bagging_common
        elif name == "ExtraTreesClassifier":
            hyperparams = rf_tree_common
        elif name == "XGBClassifier":
            hyperparams = [
                Hyperparam("n_estimators", Integer(50, 2000)),
                Hyperparam("learning_rate", Real(0.001, 0.5)),
                Hyperparam("max_depth", Integer(3, 20)),
                Hyperparam("min_child_weight", Integer(1, 20)),
                Hyperparam("subsample", Real(0.3, 1.0)),
                Hyperparam("colsample_bytree", Real(0.3, 1.0)),
                Hyperparam("gamma", Real(0.0, 10.0)),
                Hyperparam("reg_alpha", Real(1e-2, 100.0)),  # L1, wide log-scale
                Hyperparam("reg_lambda", Real(1e-2, 100.0))  # L2, wide log-scale
            ]
        elif name == "LGBMClassifier":
            hyperparams = [
                Hyperparam("n_estimators", Integer(50, 2000)),
                Hyperparam("learning_rate", Real(0.001, 0.5)),
                Hyperparam("num_leaves", Integer(20, 512)),  # More leaves for flexibility
                Hyperparam("max_depth", Integer(-1, 64)),
                Hyperparam("min_child_samples", Integer(1, 200)),
                Hyperparam("subsample", Real(0.3, 1.0)),
                Hyperparam("colsample_bytree", Real(0.3, 1.0)),
                Hyperparam("reg_alpha", Real(1e-2, 100.0)),
                Hyperparam("reg_lambda", Real(1e-2, 100.0))
            ]
        else:
            hyperparams = []

        components.append(Component(
            name=name,
            constructor=constructor,
            task=task,
            tag=Tag.MODEL_SELECTION,
            hyperparams=hyperparams
        ))

    return components


def get_null_components():
    components = []
    for name, registry_name, task, tag in null_components:
        if constructor_registry.has(registry_name):
            constructor = constructor_registry.get(registry_name)
            components.append(Component(name=name, constructor=constructor, task=task, tag=tag))
    return components

# -----------------------------
# Assemble classification search space
# -----------------------------
classification_models = get_registered_components(classification_model_names, TaskType.CLASSIFICATION)
nulls = get_null_components()

classification_space = SearchSpace(
    models=classification_models,
    data_processors=[nulls[0]],       # NoDataProcessing
    feature_processors=[nulls[1]],    # NoFeatureProcessing
    task=TaskType.CLASSIFICATION
)

# -----------------------------
# Serialize to YAML
# -----------------------------
classification_space.to_yaml(path="./custom_classification_space.yaml")
print("✅ Saved classification search space")
