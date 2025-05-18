#! /usr/bin/python3
import sys
from contextlib import redirect_stdout

import numpy as np  # Added for predict if you use argmax there
from codemaps import Codemaps  #
from dataset import Dataset  #
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Flatten,
    Conv1D,
)  # Assuming tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam  # Recommended for BERT fine-tuning

from transformers import TFBertModel

BERT_MODEL_NAME = "Linglab/PharmBERT-uncased"


def build_network(
    codes: Codemaps, learning_rate=5e-5
):  # Pass codes object for maxlen, n_labels
    """
    Builds the network with a BERT base.
    """
    n_labels = codes.get_n_labels()  #
    max_len = codes.maxlen  #

    # BERT input layers
    input_ids = Input(shape=(max_len,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype="int32", name="attention_mask")

    # BERT model
    bert_model = TFBertModel.from_pretrained(BERT_MODEL_NAME, name="bert_model")
    # Make BERT layers trainable for fine-tuning
    bert_model.trainable = True

    # Get the last hidden state from BERT
    # The output is a tuple, [0] is last_hidden_state, [1] is pooler_output
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[
        0
    ]  # Using last_hidden_state

    # Your existing CNN layers
    # You might need to adjust filter sizes or architecture if BERT's output dimension is different
    # BERT base uncased hidden size is 768. Your previous embedding dim was 100.
    # Using last_hidden_state (batch_size, max_len, hidden_size=768)

    # Option 1: Use CLS token output (representation of the whole sequence)
    # cls_token_output = bert_output[:, 0, :] # Shape: (batch_size, hidden_size)
    # x = Dropout(0.5)(cls_token_output)
    # x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x) # Example intermediate dense layer
    # x = Dropout(0.3)(x)
    # out = Dense(n_labels, activation="softmax", kernel_regularizer=l2(0.01))(x)

    # Option 2: Keep CNN layers on top of BERT's sequence output (last_hidden_state)
    # This is more similar to your original architecture's spirit.
    x = bert_output  # (batch_size, max_len, 768)

    x = Conv1D(
        filters=128, kernel_size=5, strides=1, activation="relu", padding="same"
    )(
        x
    )  # Increased filters
    x = Dropout(0.5)(x)
    x = Conv1D(filters=64, kernel_size=5, strides=1, activation="relu", padding="same")(
        x
    )  # Increased filters
    x = Dropout(0.2)(x)
    # x = Conv1D(filters=32, kernel_size=5, strides=1, activation="relu", padding="same")(x) # Potentially adjust
    # x = Dropout(0.2)(x)
    x = Flatten()(x)  #
    x = Dropout(0.5)(x)  #
    out = Dense(n_labels, activation="softmax", kernel_regularizer=l2(0.01))(x)  #

    model = Model(inputs=[input_ids, attention_mask], outputs=out)

    # Use Adam optimizer with a learning rate suitable for BERT fine-tuning
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


## --------- MAIN PROGRAM -----------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: train.py <train_data_path> <validation_data_path> <model_name_prefix>"
        )
        sys.exit(1)

    trainfile = sys.argv[1]  #
    validationfile = sys.argv[2]  #
    modelname_prefix = sys.argv[3]  # # e.g., $MODELS/bert_cnn

    # load train and validation data
    traindata = Dataset(trainfile)  #
    valdata = Dataset(validationfile)  #

    # create indexes from training data
    # This max_len is for BERT tokenizer, not words.
    max_len = 150  # As per your original max_len

    # Initialize Codemaps with BERT tokenizer settings
    codes = Codemaps(traindata, maxlen=max_len, bert_model_name=BERT_MODEL_NAME)  #

    # Resize BERT model's token embeddings if new special tokens were added
    # This is important if '<DRUG1>', etc. were not in PharmBERT's original vocab
    temp_bert_model_for_resize = TFBertModel.from_pretrained(BERT_MODEL_NAME)
    temp_bert_model_for_resize.resize_token_embeddings(len(codes.tokenizer))

    # build network
    # Consider a smaller learning rate for fine-tuning BERT, e.g., 2e-5, 3e-5, 5e-5
    model = build_network(codes, learning_rate=3e-5)  #
    with redirect_stdout(sys.stderr):  #
        model.summary()  #

    # encode datasets
    # encode_texts returns a dict: {'input_ids': ..., 'attention_mask': ...}
    Xt_encoded = codes.encode_texts(traindata)
    Yt = codes.encode_labels(traindata)  #

    Xv_encoded = codes.encode_texts(valdata)
    Yv = codes.encode_labels(valdata)  #

    # train model
    with redirect_stdout(sys.stderr):  #
        model.fit(
            [Xt_encoded["input_ids"], Xt_encoded["attention_mask"]],
            Yt,
            batch_size=64,  # BERT models are memory intensive, you might need to reduce batch size from 32
            epochs=3,  # Fine-tuning BERT usually requires fewer epochs (e.g., 3-5)
            validation_data=(
                [Xv_encoded["input_ids"], Xv_encoded["attention_mask"]],
                Yv,
            ),
            verbose=1,
        )

    # save model and codemaps (which now saves maxlen, label_index, bert_model_name)
    model.save(modelname_prefix + ".keras")  #
    codes.save(modelname_prefix + ".idx")  #

    print(f"Model saved to {modelname_prefix}.keras")
    print(f"Codemaps saved to {modelname_prefix}.idx")
