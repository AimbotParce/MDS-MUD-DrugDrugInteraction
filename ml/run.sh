#! /bin/bash

BASEDIR=..

mkdir -p $BASEDIR/data/cache
mkdir -p $BASEDIR/data/models
mkdir -p $BASEDIR/data/out

# extract features
echo "Extracting features"
python3 extract-features.py $BASEDIR/data/devel/ > $BASEDIR/data/cache/devel.cod &
python3 extract-features.py $BASEDIR/data/train/ | tee $BASEDIR/data/cache/train.cod | cut -f4- > $BASEDIR/data/cache/train.cod.cl

# train model
echo "Training model"
python3 train-sklearn.py $BASEDIR/data/models/model.joblib $BASEDIR/data/models/vectorizer.joblib < $BASEDIR/data/cache/train.cod.cl
# run model
echo "Running model..."
python3 predict-sklearn.py $BASEDIR/data/models/model.joblib $BASEDIR/data/models/vectorizer.joblib < $BASEDIR/data/cache/devel.cod > $BASEDIR/data/out/devel.out
# evaluate results
echo "Evaluating results..."
python3 evaluator.py DDI $BASEDIR/data/devel/ $BASEDIR/data/out/devel.out > $BASEDIR/data/out/devel.stats

