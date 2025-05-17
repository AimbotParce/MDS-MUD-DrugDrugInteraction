#!/usr/bin/env python3

import sys
import json
import optuna
import numpy as np
from joblib import dump
from typing import Any, Dict, List
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier, Pool


def load_data(data):
    features: List[Dict[str, Any]] = []
    labels: List[str] = []
    for interaction in data:
        label, feature_json = interaction.strip().split("\t")
        features.append(json.loads(feature_json))
        labels.append(label)
    return features, labels


def objective(trial, X, y, classes):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Plain", "Ordered"]),
        "task_type": "GPU",
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "verbose": False,
        "random_seed": 42,
    }

    clf = CatBoostClassifier(**params)
    # catboost can use Pool for features + labels, but sklearn cross_val_score needs matrix + labels
    # We'll just pass X and y directly

    score = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted").mean()
    return score


if __name__ == "__main__":
    model_file = sys.argv[1]
    vectorizer_file = sys.argv[2]

    features, labels = load_data(sys.stdin)
    y = np.asarray(labels)
    classes = np.unique(y)

    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(features)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, classes), n_trials=100)

    best_params = study.best_params
    best_params.update({
        "task_type": "GPU",
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "verbose": False,
        "random_seed": 42,
    })

    clf = CatBoostClassifier(**best_params)
    clf.fit(X, y)

    dump(clf, model_file)
    dump(vectorizer, vectorizer_file)

    print("Best params:", best_params)
