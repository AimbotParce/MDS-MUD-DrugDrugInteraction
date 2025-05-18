#!/usr/bin/env python3
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from codemaps import Codemaps
from dataset import Dataset as RawDataset


class DDIDataset(TorchDataset):
    def __init__(self, raw_dataset, codes, tokenizer, max_len):
        self.samples = list(raw_dataset.sentences())
        self.codes = codes
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = " ".join([t["form"] for t in s["sent"]])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        label = self.codes.label2idx(s["type"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }


class BERT_CNN(nn.Module):  # Kept name for compatibility, but internals changed
    def __init__(
        self,
        pretrained_model_name,
        num_labels,
        dropout_rate=0.2,  # Adjusted dropout for the new head
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        # Use BERT's hidden_size for the classifier input
        bert_hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(bert_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler_output for classification tasks
        # The pooler_output is the [CLS] token's last hidden state,
        # further processed by a Linear layer and Tanh activation.
        pooled_output = outputs.pooler_output

        # Apply dropout before the classification layer
        x = self.dropout(pooled_output)

        # Classification layer
        logits = self.classifier(x)
        return logits


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)

    train_raw = RawDataset(args.train_file)
    val_raw = RawDataset(args.val_file)

    max_len = args.max_len
    codes = Codemaps(train_raw, max_len)  # Primarily for labels and max_len
    num_labels = codes.get_n_labels()

    model_name = "Lianglab/PharmBERT-uncased"  # Or user-specified via args if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = ["<DRUG1>", "<DRUG2>", "<DRUG_OTHER>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = BERT_CNN(model_name, num_labels, dropout_rate=args.dropout_rate)
    model.to(device)
    # Resize token embeddings to account for new special tokens
    model.bert.resize_token_embeddings(len(tokenizer))

    train_ds = DDIDataset(train_raw, codes, tokenizer, max_len)
    val_ds = DDIDataset(val_raw, codes, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Calculate class weights for handling imbalance
    train_labels = [codes.label2idx(s["type"]) for s in train_raw.sentences()]
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=np.array(train_labels),
    )
    class_weights = torch.tensor(class_weights_array, dtype=torch.float).to(device)
    print(f"Calculated class weights: {class_weights}", file=sys.stderr)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Total number of training steps for learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    print(f"Starting training for {args.epochs} epochs...", file=sys.stderr)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_num, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )  # Gradient clipping
            optimizer.step()
            scheduler.step()  # Update learning rate

            total_loss += loss.item()
            if (batch_num + 1) % args.logging_steps == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs}, Batch {batch_num+1}/{len(train_loader)}, Loss: {loss.item():.4f}",
                    file=sys.stderr,
                )

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss, correct_val_preds = 0, 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(ids, mask)
                loss = criterion(
                    logits, labels
                )  # Use weighted loss for validation loss calculation too for consistency
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct_val_preds += (preds == labels).sum().item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val_preds / len(val_ds)

        # For a more detailed F1 score, you'd typically run the predict.py and evaluator.py
        # Here we just print accuracy and loss.
        print(
            f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}",
            file=sys.stderr,
        )

    # --- Saving the model ---
    out_dir = args.model_name
    os.makedirs(out_dir, exist_ok=True)

    # Save the BERT part (fine-tuned)
    model.bert.save_pretrained(out_dir)
    # Save the tokenizer
    tokenizer.save_pretrained(out_dir)

    # Save the entire model's state_dict (includes BERT fine-tuned weights + classifier head)
    # predict.py loads the BERT model from out_dir and then loads this state_dict.
    torch.save(model.state_dict(), os.path.join(out_dir, "head.pt"))

    # Save codemaps (for label to index mapping)
    codes.save(out_dir + ".idx")
    print(f"Model, tokenizer, and codemaps saved to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BERT-based model for DDI classification."
    )
    parser.add_argument(
        "train_file",
        help="Path to the training data file (parsed .pck or XML directory)",
    )
    parser.add_argument(
        "val_file",
        help="Path to the validation data file (parsed .pck or XML directory)",
    )
    parser.add_argument(
        "model_name", help="Directory name to save the trained model and tokenizer"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Batch size for training and validation",
    )  # Adjusted default
    parser.add_argument(
        "--lr", type=float, default=3e-5, help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-8, help="Epsilon for AdamW optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )  # Adjusted default
    parser.add_argument(
        "--max_len",
        type=int,
        default=150,
        help="Maximum sequence length for BERT tokenizer",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate for the classifier head",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log training loss every X batches.",
    )

    args = parser.parse_args()
    train(args)
