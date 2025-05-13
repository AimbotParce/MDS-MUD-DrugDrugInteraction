#! /usr/bin/python3
import sys
from contextlib import redirect_stdout

from codemaps import Codemaps
from dataset import Dataset
from keras import Input, layers
from keras.layers import Conv1D, Dense, Embedding, Flatten
from keras.models import Model


def add_transformer_block(embed_dim, num_heads, ff_dim, dropout, inp):
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inp, inp)
    x = layers.Dropout(dropout)(x)
    ln1 = layers.LayerNormalization(epsilon=1e-6)(inp + x)
    x = layers.Dense(ff_dim, activation="relu")(ln1)
    x = layers.Dense(embed_dim)(x)
    ffn = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(ln1 + ffn)
    return x


def build_network(idx):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # word input layer & embeddings
    inptW = Input(shape=(max_len,))
    x = Embedding(input_dim=n_words, output_dim=100, input_length=max_len, mask_zero=False)(inptW)

    x = add_transformer_block(embed_dim=100, num_heads=10, ff_dim=30, dropout=0.1, inp=x)
    x = Flatten()(x)

    x = Dense(n_labels, activation="softmax")(x)

    model = Model(inptW, x)
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
