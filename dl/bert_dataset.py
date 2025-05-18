import tensorflow as tf
from dataset import Dataset


class BERTDataset:
    def __init__(self, dataset_obj, tokenizer, max_length=150):
        self.dataset = dataset_obj
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_data(self):
        """Convert Dataset object to BERT-ready format"""
        texts = []
        labels = []

        for sentence in self.dataset.sentences():
            # Join tokens to form text, keeping special drug tokens
            sentence_text = " ".join([t["form"] for t in sentence["sent"]])
            texts.append(sentence_text)

            # Get label
            label_name = sentence["type"]
            labels.append(label_name)

        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )

        # Prepare labels (convert to one-hot)
        unique_labels = sorted(list(set(labels)))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        label_ids = [label_map[label] for label in labels]
        one_hot_labels = tf.keras.utils.to_categorical(
            label_ids, num_classes=len(unique_labels)
        )

        return encodings, one_hot_labels, label_map
