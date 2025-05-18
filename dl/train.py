#!/usr/bin/env python3
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
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


class BERT_CNN(nn.Module):
    def __init__(
        self,
        pretrained_model_name,
        num_labels,
        dropout_conv=(0.5, 0.2, 0.2),
        dropout_pool=0.5,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        self.conv1 = nn.Conv1d(hidden_size, 50, kernel_size=5, padding=2)
        self.drop1 = nn.Dropout(dropout_conv[0])
        self.conv2 = nn.Conv1d(50, 30, kernel_size=5, padding=2)
        self.drop2 = nn.Dropout(dropout_conv[1])
        self.conv3 = nn.Conv1d(30, 20, kernel_size=5, padding=2)
        self.drop3 = nn.Dropout(dropout_conv[2])
        self.drop_pool = nn.Dropout(dropout_pool)
        self.classifier = nn.Linear(20, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        x = hidden.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        x = torch.max(x, dim=2).values
        x = self.drop_pool(x)
        return self.classifier(x)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_raw = RawDataset(args.train_file)
    val_raw = RawDataset(args.val_file)
    max_len = args.max_len
    codes = Codemaps(train_raw, max_len)
    num_labels = codes.get_n_labels()
    model_name = "Lianglab/PharmBERT-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = ["<DRUG1>", "<DRUG2>", "<DRUG_OTHER>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model = BERT_CNN(model_name, num_labels)
    model.to(device)
    model.bert.resize_token_embeddings(len(tokenizer))
    train_ds = DDIDataset(train_raw, codes, tokenizer, max_len)
    val_ds = DDIDataset(val_raw, codes, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)
        model.eval()
        total_val, correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(ids, mask)
                loss = criterion(logits, labels)
                total_val += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
        avg_val = total_val / len(val_loader)
        acc = correct / len(val_ds)
        print(
            f"Epoch {epoch+1}/{args.epochs} - train_loss: {avg_train:.4f} - val_loss: {avg_val:.4f} - val_acc: {acc:.4f}",
            file=sys.stderr,
        )
    out_dir = args.model_name
    os.makedirs(out_dir, exist_ok=True)
    model.bert.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "head.pt"))
    codes.save(out_dir + ".idx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("val_file")
    parser.add_argument("model_name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=150)
    args = parser.parse_args()
    train(args)
