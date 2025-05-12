#! /usr/bin/python3
import sys
from contextlib import redirect_stdout

from codemaps import Codemaps
from dataset import Dataset
from keras import Input
from keras.layers import Conv1D, Dense, Dropout, Embedding, Flatten
from keras.models import Model
from keras.regularizers import l1, l1_l2, l2


def build_network(idx):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # word input layer & embeddings
    inp = Input(shape=(max_len,))

    x = Embedding(
        input_dim=n_words, output_dim=100, input_length=max_len, mask_zero=False, embeddings_regularizer=l2(0.01)
    )(inp)
    x = Conv1D(filters=50, kernel_size=5, strides=1, activation="relu", padding="same")(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=30, kernel_size=5, strides=1, activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=20, kernel_size=5, strides=1, activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(n_labels, activation="softmax", kernel_regularizer=l2(0.01))(x)

    model = Model(inp, x)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

## --------- MAIN PROGRAM -----------
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --


# directory with files to process
trainfile = sys.argv[1]
validationfile = sys.argv[2]
modelname = sys.argv[3]

# load train and validation data
traindata = Dataset(trainfile)
valdata = Dataset(validationfile)

# create indexes from training data
max_len = 150
suf_len = 5
codes = Codemaps(traindata, max_len)

# build network
model = build_network(codes)
with redirect_stdout(sys.stderr):
    model.summary()

# encode datasets
Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)
Xv = codes.encode_words(valdata)
Yv = codes.encode_labels(valdata)

# train model
with redirect_stdout(sys.stderr):
    model.fit(Xt, Yt, batch_size=32, epochs=10, validation_data=(Xv, Yv), verbose=1)

# save model and indexes
model.save(modelname)
codes.save(modelname + ".idx")
