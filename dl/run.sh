#! /bin/bash
BASEDIR=..

DATA=$BASEDIR/data
CACHE=$DATA/cache
MODELS=$DATA/models
OUT=$DATA/out

mkdir -p $CACHE
mkdir -p $MODELS
mkdir -p $OUT


# Check that there are arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [parse|train|predict|test]"
    exit 1
fi

if [[ "$*" == *"parse"* ]]; then
   python3 parse_data.py $DATA/train $CACHE/train.pck
   python3 parse_data.py $DATA/devel $CACHE/devel.pck
   python3 parse_data.py $DATA/test  $CACHE/test.pck
fi

if [[ "$*" == *"train"* ]]; then
    python3 train.py $CACHE/train.pck $CACHE/devel.pck $MODELS/nn
fi

if [[ "$*" == *"predict"* ]]; then
   python3 predict.py $MODELS/nn $CACHE/devel.pck $OUT/devel.out 
   python3 evaluator.py DDI $DATA/devel $OUT/devel.out | tee $OUT/devel.stats
fi

if [[ "$*" == *"test"* ]]; then
   python3 predict.py $MODELS/nn $CACHE/test.pck $OUT/test.out 
   python3 evaluator.py DDI $DATA/test $OUT/test.out | tee $OUT/test.stats
fi


