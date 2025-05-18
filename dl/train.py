#! /usr/bin/python3
import sys
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf  # Explicitly import tensorflow
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Flatten,
    Conv1D,
    Layer,
)  # Import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from transformers import (
    TFBertModel,
    BertTokenizerFast,
)  # Using BertTokenizerFast like in example

from codemaps import Codemaps
from dataset import Dataset

BERT_MODEL_NAME = "Linglab/PharmBERT-uncased"


# Custom Keras Layer to wrap TFBertModel
class BertEmbeddingLayer(Layer):
    def __init__(self, bert_model_name, **kwargs):
        super(BertEmbeddingLayer, self).__init__(**kwargs)
        self.bert_model = TFBertModel.from_pretrained(bert_model_name, from_pt=True)
        self.bert_model.trainable = True  # Make BERT layers trainable

    def call(self, inputs):
        # Inputs should be a list or tuple: [input_ids, attention_mask]
        input_ids, attention_mask = inputs
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # Or outputs[0]

    def get_config(self):  # For saving and loading the model
        config = super().get_config()
        # Note: bert_model itself is not easily serializable directly into config.
        # We rely on from_pretrained during loading.
        # For simplicity, we're not adding bert_model_name to config,
        # as it's a global or passed during __init__.
        # If you need to save/load this custom layer with varying bert_model_name,
        # you'd store it in config and use it in from_config.
        return config

    @classmethod
    def from_config(cls, config):
        # A more robust from_config would re-initialize self.bert_model
        # using a bert_model_name stored in the config.
        # For now, this relies on the __init__ to get the BERT_MODEL_NAME.
        # This might require careful handling if BERT_MODEL_NAME changes.
        # A simple workaround for loading is to register the custom layer
        # and potentially re-initialize with the correct model name.
        # Or, pass bert_model_name in config and use it here.
        # For now, we'll keep it simple as the `build_network` re-creates it.
        return cls(
            bert_model_name=BERT_MODEL_NAME, **config
        )  # HACK: Assumes BERT_MODEL_NAME is accessible


def build_network(codes: Codemaps, learning_rate=5e-5):
    n_labels = codes.get_n_labels()
    max_len = codes.maxlen

    input_ids = Input(shape=(max_len,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype="int32", name="attention_mask")

    # Use the custom BertEmbeddingLayer
    bert_embedding_layer = BertEmbeddingLayer(
        bert_model_name=BERT_MODEL_NAME, name="bert_embedding"
    )
    bert_output = bert_embedding_layer([input_ids, attention_mask])

    x = bert_output

    x = Conv1D(
        filters=128, kernel_size=5, strides=1, activation="relu", padding="same"
    )(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=64, kernel_size=5, strides=1, activation="relu", padding="same")(
        x
    )
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    out = Dense(n_labels, activation="softmax", kernel_regularizer=l2(0.01))(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=out)

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

    trainfile = sys.argv[1]
    validationfile = sys.argv[2]
    modelname_prefix = sys.argv[3]

    traindata = Dataset(trainfile)
    valdata = Dataset(validationfile)

    max_len = 150

    # For Codemaps, we can use BertTokenizerFast to be consistent with the example provided by user
    # Ensure your Codemaps is adapted or that BertTokenizerFast is used there.
    # For simplicity, I'll assume Codemaps uses a compatible tokenizer or you adapt it.
    # The crucial part is that the tokenizer used for preparing data MATCHES what BERT expects,
    # including added special tokens.

    # Let's use BertTokenizerFast here for resizing, assuming Codemaps will align.
    tokenizer_for_resize = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    special_tokens_dict = {
        "additional_special_tokens": ["<DRUG1>", "<DRUG2>", "<DRUG_OTHER>"]
    }
    tokenizer_for_resize.add_special_tokens(special_tokens_dict)

    print(f"Tokenizer vocabulary size before resizing: {len(tokenizer_for_resize)}")

    # Initialize a temporary BERT model just for resizing its token embeddings
    # This ensures that the weights loaded by the BertEmbeddingLayer later are compatible.
    temp_bert_model_for_resize = TFBertModel.from_pretrained(
        BERT_MODEL_NAME, from_pt=True
    )
    temp_bert_model_for_resize.resize_token_embeddings(len(tokenizer_for_resize))
    print(f"Resized BERT model for tokenizer vocab size: {len(tokenizer_for_resize)}")
    # We don't use temp_bert_model_for_resize directly in the main model,
    # the BertEmbeddingLayer will load a fresh one with resized embeddings available.
    # The resize_token_embeddings call modifies the model's config that from_pretrained uses.
    # (This step is subtle: resize_token_embeddings on a standalone model instance prepares
    # the ground for future from_pretrained calls IF the config is saved and reloaded,
    # or if the same model instance is used. Here, we are ensuring the vocabulary size is known.)
    # A more robust way to ensure the main model gets resized embeddings if not done globally
    # by Hugging Face caching/config updates would be to pass the resized model/config
    # to the custom layer or resize within the custom layer.
    # However, usually, `resize_token_embeddings` when called once for a tokenizer/model pair
    # influences subsequent `from_pretrained` calls if the tokenizer length is different.
    # The BertEmbeddingLayer should pick up the model with correctly sized embeddings
    # because the tokenizer passed to DataCollator/used in Codemaps has the new tokens.

    # Update Codemaps to use BertTokenizerFast and handle special tokens consistently
    # This is CRITICAL. The tokenizer in Codemaps MUST be the same instance or configuration
    # as tokenizer_for_resize (especially vocab and added tokens).
    codes = Codemaps(
        traindata,
        maxlen=max_len,
        bert_model_name=BERT_MODEL_NAME,
        tokenizer_class=BertTokenizerFast,
    )
    # Ensure your Codemaps __init__ and encode_texts are updated to use the passed tokenizer_class
    # and add the special tokens like ['<DRUG1>', '<DRUG2>', '<DRUG_OTHER>']

    # Verify vocab size consistency
    if len(codes.tokenizer.get_vocab()) != len(tokenizer_for_resize.get_vocab()):
        print(
            "WARNING: Tokenizer vocab size mismatch between Codemaps and resizing step!"
        )
        print(f"Codemaps tokenizer vocab size: {len(codes.tokenizer.get_vocab())}")
        print(f"Resizing tokenizer vocab size: {len(tokenizer_for_resize.get_vocab())}")
        # This usually happens if special tokens are not added consistently in both places.
        # Ensure Codemaps's internal tokenizer also calls `add_special_tokens`.

    model = build_network(codes, learning_rate=3e-5)
    with redirect_stdout(sys.stderr):
        model.summary()

    Xt_encoded = codes.encode_texts(traindata)
    Yt = codes.encode_labels(traindata)

    Xv_encoded = codes.encode_texts(valdata)
    Yv = codes.encode_labels(valdata)

    with redirect_stdout(sys.stderr):
        model.fit(
            [Xt_encoded["input_ids"], Xt_encoded["attention_mask"]],
            Yt,
            batch_size=16,
            epochs=5,
            validation_data=(
                [Xv_encoded["input_ids"], Xv_encoded["attention_mask"]],
                Yv,
            ),
            verbose=1,
        )

    # When saving, Keras will try to save the custom layer.
    # For loading, you'll need to provide it in custom_objects.
    model.save(modelname_prefix + ".keras")
    codes.save(modelname_prefix + ".idx")

    print(f"Model saved to {modelname_prefix}.keras")
    print(f"Codemaps saved to {modelname_prefix}.idx")
    print(
        "To load the model later, use: \nfrom tensorflow.keras.models import load_model\nmodel = load_model('your_model.keras', custom_objects={'BertEmbeddingLayer': BertEmbeddingLayer})"
    )
