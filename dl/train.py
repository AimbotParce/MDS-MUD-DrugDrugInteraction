#! /usr/bin/python3


import sys
import random
from contextlib import redirect_stdout

from tensorflow.keras import regularizers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPool1D, Reshape, Concatenate, Flatten, Bidirectional, LSTM

from dataset import *
from codemaps import *

def build_network(idx) :

   # sizes
   n_words = codes.get_n_words()
   max_len = codes.maxlen
   n_labels = codes.get_n_labels()

   # word input layer & embeddings
   inptW = Input(shape=(max_len,))
   embW = Embedding(input_dim=n_words, output_dim=100,
                      input_length=max_len, mask_zero=False)(inptW)  

   conv = Conv1D(filters=30, kernel_size=2, strides=1, activation='relu', padding='same')(embW)
   flat= Flatten()(conv)
   
   out = Dense(n_labels, activation='softmax')(flat)

   model = Model(inptW, out)
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
with redirect_stdout(sys.stderr) :
   model.summary()

# encode datasets
Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)
Xv = codes.encode_words(valdata)
Yv = codes.encode_labels(valdata)

# train model
with redirect_stdout(sys.stderr) :
   model.fit(Xt, Yt, batch_size=32, epochs=10, validation_data=(Xv,Yv), verbose=1)
   
# save model and indexs
model.save(modelname)
codes.save(modelname)

