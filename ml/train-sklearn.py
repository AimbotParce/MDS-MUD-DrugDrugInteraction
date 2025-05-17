#!/usr/bin/env python3

import argparse
import json
import sys
from typing import Any, Dict, List

import numpy as np
from joblib import dump
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB


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

    train_features, y_train = load_data(sys.stdin)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)

    v = DictVectorizer()
    print(train_features)
    X_train = v.fit_transform(train_features)

    clf = MultinomialNB(alpha=0.01)
    clf.partial_fit(X_train, y_train, classes)

    # Save classifier and DictVectorizer
    dump(clf, model_file)
    dump(v, vectorizer_file)
