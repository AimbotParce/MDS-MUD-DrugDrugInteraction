#!/usr/bin/env python3

import sys
import json
import optuna
import numpy as np
from joblib import dump
from typing import Any, Dict, List
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


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
        "objective": "multi:softprob",
        "num_class": len(classes),
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
        "tree_method": "gpu_hist",  # GPU acceleration
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
    }

    clf = XGBClassifier(**params)
    score = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted").mean()
    return score


if __name__ == "__main__":
    model_file = sys.argv[1]
    vectorizer_file = sys.argv[2]

    features, labels = load_data(sys.stdin)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)  # encode string labels to integers
    classes = le.classes_          # original class names

    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(features)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, classes), n_trials=100)  # Increase trials as needed

    best_params = study.best_params
    best_params.update({
        "objective": "multi:softprob",
        "num_class": len(classes),
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "tree_method": "gpu_hist",
    })

    clf = XGBClassifier(**best_params)
    clf.fit(X, y)

    dump(clf, model_file)
    dump(vectorizer, vectorizer_file)

    print("Best params:", best_params)
