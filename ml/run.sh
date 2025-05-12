#! /bin/bash

BASEDIR=..
DATA=$BASEDIR/data
CACHE=$DATA/cache
MODELS=$DATA/models
OUT=$DATA/out

mkdir -p $CACHE
mkdir -p $MODELS
mkdir -p $OUT

# extract features
echo "Extracting features"
python3 extract-features.py $DATA/devel/ > $CACHE/devel.cod &
python3 extract-features.py $DATA/train/ | tee $CACHE/train.cod | cut -f4- > $CACHE/train.cod.cl

# train model
echo "Training model"
python3 train-sklearn.py $MODELS/model.joblib $MODELS/vectorizer.joblib < $CACHE/train.cod.cl
# run model
echo "Running model..."
python3 predict-sklearn.py $MODELS/model.joblib $MODELS/vectorizer.joblib < $CACHE/devel.cod > $OUT/devel.out
# evaluate results
echo "Evaluating results..."
python3 evaluator.py DDI $DATA/devel/ $OUT/devel.out > $OUT/devel.stats

