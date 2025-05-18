#! /usr/bin/python3

import sys
from os import system

from codemaps import *
from dataset import *
from keras.models import Model, load_model

from transformers import AutoTokenizer
from bert_model import DDIBertModel

## --------- Entity extractor -----------
## -- Extract drug entities from given text and return them as
## -- a list of dictionaries with keys "offset", "text", and "type"


def output_interactions(data, preds, outfile):

    # print(testdata[0])
    outf = open(outfile, "w")
    for exmp, tag in zip(data.sentences(), preds):
        sid = exmp["sid"]
        e1 = exmp["e1"]
        e2 = exmp["e2"]
        if tag != "null":
            print(sid, e1, e2, tag, sep="|", file=outf)

    outf.close()


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir
## --

fname = sys.argv[1]
datafile = sys.argv[2]
outfile = sys.argv[3]

is_bert_model = os.path.exists(fname + "_tokenizer")

if is_bert_model:
    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(fname + "_tokenizer")

    # Load BERT model
    model = load_model(fname)

    # Load test data
    testdata = Dataset(datafile)

    # Process sentences
    texts = []
    for sentence in testdata.sentences():
        sentence_text = " ".join([t["form"] for t in sentence["sent"]])
        texts.append(sentence_text)

    # Tokenize texts
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=200,  # Use the same max_length as in training
        return_tensors="tf",
    )

    # Make predictions
    predictions = model.predict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
    )

    # Convert predictions to labels
    codes = Codemaps(fname + ".idx")
    predicted_labels = [codes.idx2label(np.argmax(pred)) for pred in predictions]

    # Extract relations
    output_interactions(testdata, predicted_labels, outfile)
else:
    model = load_model(fname)
    codes = Codemaps(fname + ".idx")

    testdata = Dataset(datafile)
    X = codes.encode_words(testdata)

    Y = model.predict(X)
    Y = [codes.idx2label(np.argmax(s)) for s in Y]

    # extract relations
    output_interactions(testdata, Y, outfile)
