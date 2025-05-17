#!/bin/bash

set -e
set -u
set -o pipefail

BASEDIR=..

DATA=$BASEDIR/data
CACHE=$DATA/cache
MODELS=$DATA/models
OUT=$DATA/out

mkdir -p $CACHE
mkdir -p $MODELS
mkdir -p $OUT

# train model
echo "Training model"
python3 train-sklearn.py \
    $MODELS/model.joblib \
    $MODELS/vectorizer.joblib \
    $MODELS/labelencoder.joblib \
    < $CACHE/train.cod.cl

# run model
echo "Running model..."
python3 predict-sklearn.py \
    $MODELS/model.joblib \
    $MODELS/vectorizer.joblib \
    $MODELS/labelencoder.joblib \
    < $CACHE/devel.cod > $OUT/devel.out

# evaluate results
echo "Evaluating results..."
python3 evaluator.py DDI $DATA/devel/ $OUT/devel.out > $OUT/devel.stats
