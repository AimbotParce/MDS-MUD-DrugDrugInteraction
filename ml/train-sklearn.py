#!/usr/bin/env python3

import sys
import json
import optuna
import numpy as np
from joblib import dump
from typing import Any, Dict, List
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
import lightgbm as lgb


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
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "device": "cpu",  # <--- enable GPU, however one must install lightgbm from source not from pip install (only CPU)
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
    }

    clf = lgb.LGBMClassifier(**params)
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
    study.optimize(lambda trial: objective(trial, X, y, classes), n_trials=50)

    best_params = study.best_params
    best_params.update({
        "objective": "multiclass",
        "num_class": len(classes),
        "verbosity": -1
    })

    clf = lgb.LGBMClassifier(**best_params)
    clf.fit(X, y)

    dump(clf, model_file)
    dump(vectorizer, vectorizer_file)

    print("Best params:", best_params)
