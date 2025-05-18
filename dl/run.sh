#! /bin/bash

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

if [[ "$*" == *"bert_train"* ]]; then
    python3 train.py $CACHE/train.pck $CACHE/devel.pck $MODELS/bert_nn.keras
fi

if [[ "$*" == *"predict"* ]]; then
   python3 predict.py $MODELS/nn.keras $CACHE/devel.pck $OUT/devel.out 
   python3 evaluator.py DDI $DATA/devel $OUT/devel.out | tee $OUT/devel.stats
fi

if [[ "$*" == *"bert_test"* ]]; then
   python3 predict.py $MODELS/bert_nn.keras $CACHE/test.pck $OUT/bert_test.out 
   python3 evaluator.py DDI $DATA/test $OUT/bert_test.out | tee $OUT/bert_test.stats
fi




