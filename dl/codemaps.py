from typing import Dict, Union

import numpy as np
from dataset import Dataset
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class Codemaps:
    """
    Class to create and manage codemaps for words, lemmas, PoS tags and labels
    in a dataset. The codemaps are used to encode the words, lemmas, PoS tags and labels
    as numbers when training the neural network. The codemaps are created from the training
    data and saved to a file. The codemaps can be loaded from a file when training the
    neural network.
    """

    def __init__(self, data: Union[Dataset, str], maxlen: int = None):
        """
        Constructor for the Codemaps class. It creates the codemaps from the training data
        and saves them to a file. The codemaps are used to encode the words, lemmas, PoS tags
        and labels as numbers when training the neural network. The codemaps are created from
        the training data and saved to a file. The codemaps can be loaded from a file when
        training the neural network.

        Args:
            data (Dataset | str): Either a Dataset object containing the training data or a string
                containing the name of the file to load the codemaps from.
            maxlen (int, optional): The maximum length of the sentences. It can only be None
                if data is a string. If data is a Dataset object, maxlen must be an integer.
                Defaults to None.
        """

        self.maxlen: int
        self.word_index: Dict[str, int]
        self.lc_word_index: Dict[str, int]
        self.lemma_index: Dict[str, int]
        self.pos_index: Dict[str, int]
        self.label_index: Dict[str, int]

        if isinstance(data, Dataset) and maxlen is not None:
            self._create_indexes(data, maxlen)

        elif type(data) == str and maxlen is None:
            self._load_indexes(data)

        else:
            print("codemaps: Invalid or missing parameters in constructor")
            exit(1)

    def _create_indexes(self, data: Dataset, maxlen: int):
        """
        Extract all words and labels in given sentences and
        create indexes to encode them as numbers when needed
        """

        self.maxlen = maxlen
        words = set()
        lc_words = set()
        lems = set()
        pos = set()
        labels = set()

        for s in data.sentences():
            for t in s["sent"]:
                words.add(t["form"])
                lc_words.add(t["lc_form"])
                lems.add(t["lemma"])
                pos.add(t["pos"])
            labels.add(s["type"])

        self.word_index = {w: i + 2 for i, w in enumerate(sorted(list(words)))}
        self.word_index["PAD"] = 0  # Padding
        self.word_index["UNK"] = 1  # Unknown words

        self.lc_word_index = {w: i + 2 for i, w in enumerate(sorted(list(lc_words)))}
        self.lc_word_index["PAD"] = 0  # Padding
        self.lc_word_index["UNK"] = 1  # Unknown words

        self.lemma_index = {s: i + 2 for i, s in enumerate(sorted(list(lems)))}
        self.lemma_index["PAD"] = 0  # Padding
        self.lemma_index["UNK"] = 1  # Unseen lemmas

        self.pos_index = {s: i + 2 for i, s in enumerate(sorted(list(pos)))}
        self.pos_index["PAD"] = 0  # Padding
        self.pos_index["UNK"] = 1  # Unseen PoS tags

        self.label_index = {t: i for i, t in enumerate(sorted(list(labels)))}

    def _load_indexes(self, name):
        """
        Load indexes from file
        """
        self.maxlen = 0
        self.word_index = {}
        self.lc_word_index = {}
        self.lemma_index = {}
        self.pos_index = {}
        self.label_index = {}

        with open(name + ".idx") as f:
            for line in f.readlines():
                (t, k, i) = line.split()
                if t == "MAXLEN":
                    self.maxlen = int(k)
                elif t == "WORD":
                    self.word_index[k] = int(i)
                elif t == "LCWORD":
                    self.lc_word_index[k] = int(i)
                elif t == "LEMMA":
                    self.lemma_index[k] = int(i)
                elif t == "POS":
                    self.pos_index[k] = int(i)
                elif t == "LABEL":
                    self.label_index[k] = int(i)

    ## ---------- Save model and indexs ---------------
    def save(self, name: str):
        # save indexes
        with open(name + ".idx", "w") as f:
            print("MAXLEN", self.maxlen, "-", file=f)
            for key in self.label_index:
                print("LABEL", key, self.label_index[key], file=f)
            for key in self.word_index:
                print("WORD", key, self.word_index[key], file=f)
            for key in self.lc_word_index:
                print("LCWORD", key, self.lc_word_index[key], file=f)
            for key in self.lemma_index:
                print("LEMMA", key, self.lemma_index[key], file=f)
            for key in self.pos_index:
                print("POS", key, self.pos_index[key], file=f)

    @staticmethod
    def _get_code(index: Dict[str, int], k: str):
        """
        Get code for key k in given index, or code for unknown if not found
        """
        return index[k] if k in index else index["UNK"]

    def _encode_and_pad(self, data: Dataset, index: Dict[str, int], key: str):
        """
        Encode and pad all sequences of given key (form, lemma, etc)
        """
        X = [[self._get_code(index, w[key]) for w in s["sent"]] for s in data.sentences()]
        X = pad_sequences(maxlen=self.maxlen, sequences=X, padding="post", value=index["PAD"])
        return X

    def encode_words(self, data: Dataset):
        """
        Encode X from given data
        """

        # encode and pad sentence words
        Xw = self._encode_and_pad(data, self.word_index, "form")
        # encode and pad sentence lc_words
        Xlw = self._encode_and_pad(data, self.lc_word_index, "lc_form")
        # encode and pad lemmas
        Xl = self._encode_and_pad(data, self.lemma_index, "lemma")
        # encode and pad PoS
        Xp = self._encode_and_pad(data, self.pos_index, "pos")

        # return encoded sequences
        # return [Xw,Xlw,Xl,Xp] (or just the subset expected by the NN inputs)
        return Xw

    ## --------- encode Y from given data -----------
    def encode_labels(self, data: Dataset):
        # encode and pad sentence labels
        Y = [self.label_index[s["type"]] for s in data.sentences()]
        Y = [to_categorical(i, num_classes=self.get_n_labels()) for i in Y]
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self):
        return len(self.word_index)

    ## -------- get word index size ---------
    def get_n_lc_words(self):
        return len(self.lc_word_index)

    ## -------- get label index size ---------
    def get_n_labels(self):
        return len(self.label_index)

    ## -------- get label index size ---------
    def get_n_lemmas(self):
        return len(self.lemma_index)

    ## -------- get label index size ---------
    def get_n_pos(self):
        return len(self.pos_index)

    ## -------- get index for given word ---------
    def word2idx(self, w: str):
        return self.word_index[w]

    ## -------- get index for given word ---------
    def lcword2idx(self, w: str):
        return self.lc_word_index[w]

    ## -------- get index for given label --------
    def label2idx(self, l: str):
        return self.label_index[l]

    ## -------- get label name for given index --------
    def idx2label(self, i: int):
        for l in self.label_index:
            if self.label_index[l] == i:
                return l
        raise KeyError
