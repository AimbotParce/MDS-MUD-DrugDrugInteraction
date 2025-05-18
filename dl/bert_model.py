import tensorflow as tf
from tensorflow.keras import layers, Model, mixed_precision
from transformers import TFAutoModel, AutoTokenizer


class DDIBertModel:
    def __init__(
        self, bert_model_name="Lianglab/PharmBERT", max_length=150, num_labels=5
    ):
        self.bert_model_name = bert_model_name
        self.max_length = max_length
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        # Add special tokens for drug entities
        special_tokens = {
            "additional_special_tokens": ["<DRUG1>", "<DRUG2>", "<DRUG_OTHER>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)

    def build_model(self):
        # Use fp16 to avoid running out of vram
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

        # Input layers
        input_ids = layers.Input(
            shape=(self.max_length,), dtype=tf.int32, name="input_ids"
        )
        attention_mask = layers.Input(
            shape=(self.max_length,), dtype=tf.int32, name="attention_mask"
        )

        # Load PharmBERT model
        bert_model = TFAutoModel.from_pretrained(self.bert_model_name)
        bert_model.resize_token_embeddings(len(self.tokenizer))

        # Get BERT embeddings
        bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs[0][:, 0, :]  # [CLS] token representation

        # Add dropout for regularization
        x = layers.Dropout(0.3)(pooled_output)

        # Add a dense layer before final classification
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

        # Classification layer
        outputs = layers.Dense(self.num_labels, activation="softmax")(x)

        # Define model
        model = Model(inputs=[input_ids, attention_mask], outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        # Compile model
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        return model

    def tokenize_text(self, sentences):
        """Tokenize a list of sentences for BERT input"""
        encodings = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )
        return encodings
