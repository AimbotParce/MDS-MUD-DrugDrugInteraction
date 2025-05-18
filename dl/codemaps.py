from typing import Dict, Union, List, Type  # Import Type
import numpy as np
from dataset import Dataset
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer, BertTokenizerFast  # Add BertTokenizerFast


class Codemaps:
    def __init__(
        self,
        data: Union[Dataset, str],
        maxlen: int = None,
        bert_model_name: str = "Linglab/PharmBERT-uncased",
        tokenizer_class: Type[
            Union[BertTokenizer, BertTokenizerFast]
        ] = BertTokenizerFast,
    ):  # Allow specifying tokenizer
        self.maxlen: int
        self.label_index: Dict[str, int]
        self.tokenizer: Union[BertTokenizer, BertTokenizerFast]  # Type hint
        self.bert_model_name: str = bert_model_name
        self._tokenizer_class = tokenizer_class  # Store tokenizer class

        # Define special tokens to be added
        self.special_tokens_list = ["<DRUG1>", "<DRUG2>", "<DRUG_OTHER>"]

        if isinstance(data, Dataset) and maxlen is not None:
            self.maxlen = maxlen
            self.tokenizer = self._tokenizer_class.from_pretrained(self.bert_model_name)
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": self.special_tokens_list}
            )
            self._create_label_index(data)

        elif isinstance(data, str) and maxlen is None:  # Loading from file
            self._load_indexes(data)
            # Re-initialize tokenizer after loading
            self.tokenizer = self._tokenizer_class.from_pretrained(self.bert_model_name)
            # Important: ensure special tokens are re-added if not part of base vocab
            # or if tokenizer doesn't save this info with from_pretrained.
            # add_special_tokens usually returns the number of tokens added.
            num_added_toks = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": self.special_tokens_list}
            )
            if num_added_toks > 0:
                print(
                    f"Re-added {num_added_toks} special tokens to tokenizer in Codemaps during load."
                )
        else:
            print("codemaps: Invalid or missing parameters in constructor")
            exit(1)

    def _create_label_index(self, data: Dataset):
        labels = set()
        for s in data.sentences():
            labels.add(s["type"])
        self.label_index = {t: i for i, t in enumerate(sorted(list(labels)))}

    def _load_indexes(self, name: str):
        self.maxlen = 0
        self.label_index = {}
        temp_bert_model_name = self.bert_model_name
        # Default tokenizer class if not saved (for backward compatibility if you add it later)
        temp_tokenizer_class_name = self._tokenizer_class.__name__

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
                elif t == "TOKENIZER_CLASS":  # Save tokenizer class name
                    temp_tokenizer_class_name = k

        self.bert_model_name = temp_bert_model_name
        if temp_tokenizer_class_name == "BertTokenizerFast":
            self._tokenizer_class = BertTokenizerFast
        else:  # Default or specific handling
            self._tokenizer_class = BertTokenizer

    def save(self, name: str):
        with open(name, "w") as f:
            print("MAXLEN", self.maxlen, "-", file=f)
            print("BERT_MODEL_NAME", self.bert_model_name, "-", file=f)
            print(
                "TOKENIZER_CLASS", self._tokenizer_class.__name__, "-", file=f
            )  # Save tokenizer class name
            for key in self.label_index:
                print("LABEL", key, self.label_index[key], file=f)

    def encode_texts(self, data: Dataset):
        sentences = [
            " ".join([token["form"] for token in s["sent"]]) for s in data.sentences()
        ]
        encoded_inputs = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            max_length=self.maxlen,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="tf",
        )
        return {
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
        }

    def encode_labels(self, data: Dataset):
        Y = [self.label_index[s["type"]] for s in data.sentences()]
        Y = [to_categorical(i, num_classes=self.get_n_labels()) for i in Y]
        return np.array(Y)

    def get_n_labels(self):
        return len(self.label_index)

    def label2idx(self, l: str):
        return self.label_index[l]

    def idx2label(self, i: int):
        for l_key, l_val in self.label_index.items():
            if l_val == i:
                return l_key
        raise KeyError(f"Index {i} not found in label_index")
