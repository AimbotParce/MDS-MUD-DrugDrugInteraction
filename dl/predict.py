#! /usr/bin/python3

import sys
import numpy as np
from codemaps import Codemaps  #
from dataset import Dataset  #
from tensorflow.keras.models import load_model  # Assuming tf.keras
from transformers import (
    TFBertModel,
)  # Needed for custom objects if not automatically handled

BERT_MODEL_NAME = "Linglab/PharmBERT-uncased"


def output_interactions(data, preds_indices, codes_obj, outfile):  #
    outf = open(outfile, "w")  #
    for exmp, pred_idx in zip(data.sentences(), preds_indices):  #
        sid = exmp["sid"]  #
        e1 = exmp["e1"]  #
        e2 = exmp["e2"]  #
        tag = codes_obj.idx2label(
            pred_idx
        )  # Use codes object to convert index to label
        if tag != "null":  #
            print(sid, e1, e2, tag, sep="|", file=outf)  #
    outf.close()  #


## --------- MAIN PROGRAM -----------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: predict.py <model_path.keras> <datafile_path> <output_file_path>")
        sys.exit(1)

    model_path = sys.argv[1]  #
    datafile = sys.argv[2]  #
    outfile = sys.argv[3]  #

    # When loading the model, Keras needs to know about the custom TFBertModel layer.
    # Often, this is handled automatically if you saved a TensorFlow Keras model.
    # If not, you might need: custom_objects={'TFBertModel': TFBertModel}
    try:
        model = load_model(model_path, custom_objects={"TFBertModel": TFBertModel})  #
    except Exception as e:
        print(f"Error loading model normally: {e}")
        print(
            "Attempting to load model without custom_objects dictionary (might fail if TFBertModel is not registered)."
        )
        model = load_model(model_path)

    # Load Codemaps. It will initialize tokenizer based on saved bert_model_name or default.
    # Ensure the .idx file is in the same location and has the same prefix as the model.
    codes = Codemaps(model_path.replace(".keras", ".idx"))  #
    # Important: Ensure the tokenizer used for prediction is identical to the one used for training.
    # This includes any added special tokens. The Codemaps class should handle this.

    testdata = Dataset(datafile)  #

    # Encode data using BERT tokenizer via Codemaps
    X_encoded = codes.encode_texts(testdata)  #

    # Predict
    # The model expects a list or dict of inputs based on how it was defined.
    # Our model: Model(inputs=[input_ids, attention_mask], outputs=out)
    Y_pred_probs = model.predict(
        [X_encoded["input_ids"], X_encoded["attention_mask"]]
    )  #
    Y_pred_indices = np.argmax(
        Y_pred_probs, axis=1
    )  # # Get the index of the max probability

    # Output interactions
    output_interactions(testdata, Y_pred_indices, codes, outfile)  #
    print(f"Predictions saved to {outfile}")
