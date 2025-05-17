#!/usr/bin/env python3

import sys
import json
import numpy as np
from joblib import dump
from typing import Any, Dict, List
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from ezautoml.model import eZAutoML
from ezautoml.space.search_space import SearchSpace
from ezautoml.evaluation.metric import MetricSet, Metric
from ezautoml.evaluation.task import TaskType

def load_data(data):
    features: List[Dict[str, Any]] = []
    labels: List[str] = []
    for interaction in data:
        label, feature_json = interaction.strip().split("\t")
        features.append(json.loads(feature_json))
        labels.append(label)
    return features, labels

if __name__ == "__main__":
    model_file = sys.argv[1]
    vectorizer_file = sys.argv[2]

    # Load raw data from stdin
    features, labels = load_data(sys.stdin)
    y = np.asarray(labels)

    # Vectorize features
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(features)

    # Train/test split — very important!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Define weighted F1 metric for multi-class classification
    metrics = MetricSet(
        {"f1_weighted": Metric(
            name="f1_weighted",
            fn=lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
            minimize=False
        )},
        primary_metric_name="f1_weighted"
    )

    # Load built-in classification search space
    search_space = SearchSpace.from_builtin("classification_space")

    # Instantiate ezAutoML with desired settings
    automl = eZAutoML(
        search_space=search_space,
        task=TaskType.CLASSIFICATION,
        metrics=metrics,
        max_trials=100,       # run 100 trials
        max_time=3600,        # or max 1 hour
        seed=42,
        verbose=True
    )

    # Fit on training data
    automl.fit(X_train, y_train)

    # Evaluate on test data
    test_score = automl.test(X_test, y_test)

    # Print summary of top models
    automl.summary(k=10)

    print(f"Best model test f1_weighted: {test_score:.4f}")

    # Save the best model and vectorizer for future inference
    dump(automl.best_model, model_file)
    dump(vectorizer, vectorizer_file)
