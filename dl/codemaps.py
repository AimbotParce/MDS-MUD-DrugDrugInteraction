from typing import Dict, Union, List

import numpy as np
from dataset import Dataset  #

# from keras.preprocessing.sequence import pad_sequences # Not needed for BERT tokenization like this
from tensorflow.keras.utils import to_categorical  # Assuming tf.keras
from transformers import BertTokenizer


class Codemaps:
    """
    Class to manage BERT tokenization for input sentences and encode labels.
    """

    def __init__(
        self,
        data: Union[Dataset, str],
        maxlen: int = None,
        bert_model_name: str = "Linglab/PharmBERT-uncased",
    ):
        """
        Constructor for the Codemaps class.
        If data is a Dataset object, it initializes the tokenizer and label_index.
        If data is a string (path to saved codemaps), it loads them.

        Args:
            data (Dataset | str): Either a Dataset object or a path to load codemaps.
            maxlen (int, optional): The maximum length of the tokenized sentences.
                                    Required if data is a Dataset object.
            bert_model_name (str): Name of the pre-trained BERT model for the tokenizer.
        """

        self.maxlen: int
        self.label_index: Dict[str, int]
        self.tokenizer: BertTokenizer
        self.bert_model_name: str = bert_model_name

        if isinstance(data, Dataset) and maxlen is not None:
            self.maxlen = maxlen
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
            # Add special drug tokens if they are not already part of the tokenizer's vocabulary
            # These tokens are used in your dataset.py file
            special_tokens_dict = {
                "additional_special_tokens": ["<DRUG1>", "<DRUG2>", "<DRUG_OTHER>"]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)

            self._create_label_index(data)

        elif isinstance(data, str) and maxlen is None:  # Loading from file
            self._load_indexes(data)
            # Initialize tokenizer after loading, bert_model_name should be saved or passed
            # For simplicity, we'll re-initialize it here. Ideally, save/load tokenizer info.
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
            special_tokens_dict = {
                "additional_special_tokens": ["<DRUG1>", "<DRUG2>", "<DRUG_OTHER>"]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)

        else:
            print("codemaps: Invalid or missing parameters in constructor")
            exit(1)

    def _create_label_index(self, data: Dataset):
        labels = set()
        for s in data.sentences():
            labels.add(s["type"])
        self.label_index = {t: i for i, t in enumerate(sorted(list(labels)))}

    def _load_indexes(self, name: str):
        """
        Load maxlen and label_index from file.
        Tokenizer is re-initialized based on self.bert_model_name.
        """
        self.maxlen = 0
        self.label_index = {}

        # Try to load bert_model_name if saved, otherwise use default
        temp_bert_model_name = self.bert_model_name

        with open(name) as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts:
                    continue

                t = parts[0]
                k = parts[1]

                if t == "MAXLEN":
                    self.maxlen = int(k)
                elif t == "LABEL":
                    self.label_index[k] = int(parts[2])
                elif t == "BERT_MODEL_NAME":
                    temp_bert_model_name = k

        self.bert_model_name = temp_bert_model_name

    def save(self, name: str):
        # save maxlen and label_index
        with open(name, "w") as f:
            print("MAXLEN", self.maxlen, "-", file=f)
            print(
                "BERT_MODEL_NAME", self.bert_model_name, "-", file=f
            )  # Save model name
            for key in self.label_index:
                print("LABEL", key, self.label_index[key], file=f)

    def encode_texts(self, data: Dataset):
        """
        Encode sentences from data using the BERT tokenizer.
        Returns a dictionary of input_ids and attention_mask.
        """
        sentences = [
            " ".join([token["form"] for token in s["sent"]]) for s in data.sentences()
        ]

        encoded_inputs = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            max_length=self.maxlen,
            padding="max_length",  # or True
            truncation=True,
            return_attention_mask=True,
            return_tensors="tf",  # Return TensorFlow tensors
        )
        return {
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
        }

    def encode_labels(self, data: Dataset):  #
        Y = [self.label_index[s["type"]] for s in data.sentences()]  #
        Y = [to_categorical(i, num_classes=self.get_n_labels()) for i in Y]  #
        return np.array(Y)  #

    def get_n_labels(self):  #
        return len(self.label_index)  #

    def label2idx(self, l: str):  #
        return self.label_index[l]  #

    def idx2label(self, i: int):  #
        for l_key, l_val in self.label_index.items():  #
            if l_val == i:  #
                return l_key  #
        raise KeyError(f"Index {i} not found in label_index")
