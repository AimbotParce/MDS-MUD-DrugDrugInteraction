#!/usr/bin/env python3
"""
Fine-tune BERT for Drug-Drug Interaction (DDI) classification.
Usage:
  python train_bert.py <train_data_dir> <dev_data_dir> <output_dir> [--model_name MODEL] [--epochs N] [--batch_size N]
Example:
  python train_bert.py ../data/train ../data/devel model_bert --model_name bert-base-uncased --epochs 3 --batch_size 16
"""
import os
import argparse
import json
from xml.dom import minidom

import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


def parse_ddi_data(data_dir):
    """
    Parse DDI XML files in a directory.
    Returns lists of texts (with entity markers) and labels, and the sorted set of labels.
    """
    texts = []
    labels = []
    label_set = set()
    # process only XML files
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".xml"):
            continue
        path = os.path.join(data_dir, fname)
        doc = minidom.parse(path)
        # iterate sentences
        for s in doc.getElementsByTagName("sentence"):
            stext = s.getAttribute("text")
            # collect entities by id
            ents = {e.getAttribute("id"): e for e in s.getElementsByTagName("entity")}
            # iterate pairs
            for p in s.getElementsByTagName("pair"):
                ddi = p.getAttribute("ddi")
                label = p.getAttribute("type") if ddi == "true" else "null"
                label_set.add(label)
                e1 = p.getAttribute("e1")
                e2 = p.getAttribute("e2")
                # mark entities in sentence text
                # wrap only first occurrence to avoid accidental replacements
                ent1_text = ents[e1].getAttribute("text")
                ent2_text = ents[e2].getAttribute("text")
                marked = stext.replace(ent1_text, f"[E1] {ent1_text} [/E1]", 1).replace(
                    ent2_text, f"[E2] {ent2_text} [/E2]", 1
                )
                texts.append(marked)
                labels.append(label)
    return texts, labels, sorted(label_set)


class DDIDataset(Dataset):
    """PyTorch dataset for tokenized DDI examples."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", help="Directory with training XML files")
    parser.add_argument("dev_dir", help="Directory with development XML files")
    parser.add_argument(
        "output_dir", help="Where to save the fine-tuned model and mappings"
    )
    parser.add_argument(
        "--model_name", default="bert-base-uncased", help="Pre-trained BERT model"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size per device"
    )
    args = parser.parse_args()

    # prepare output
    os.makedirs(args.output_dir, exist_ok=True)

    # load and parse data
    train_texts, train_labels, label_list = parse_ddi_data(args.train_dir)
    dev_texts, dev_labels, _ = parse_ddi_data(args.dev_dir)

    # label mapping
    label2id = {label: idx for idx, label in enumerate(label_list)}
    train_labels = [label2id[l] for l in train_labels]
    dev_labels = [label2id[l] for l in dev_labels]

    # tokenizer and special tokens for entity markers
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)

    # tokenize datasets
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, return_tensors="pt"
    )
    dev_encodings = tokenizer(
        dev_texts, truncation=True, padding=True, return_tensors="pt"
    )

    # wrap into PyTorch datasets
    train_dataset = DDIDataset(train_encodings, train_labels)
    dev_dataset = DDIDataset(dev_encodings, dev_labels)

    # load pre-trained model and adjust for new tokens
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label_list)
    )
    model.resize_token_embeddings(len(tokenizer))

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # simple accuracy metric
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = (preds == labels).astype(float).mean().item()
        return {"accuracy": acc}

    # data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # train and save
    trainer.train()
    trainer.save_model(args.output_dir)
    # save label mappings
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)
    with open(os.path.join(args.output_dir, "id2label.json"), "w") as f:
        json.dump({v: k for k, v in label2id.items()}, f, indent=2)


if __name__ == "__main__":
    main()
