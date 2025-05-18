#!/usr/bin/env python3
import sys
import os
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer
from codemaps import Codemaps
from dataset import Dataset as RawDataset
from train import BERT_CNN

class DDIDataset(TorchDataset):
    def __init__(self, raw_dataset, tokenizer, max_len):
        self.samples = list(raw_dataset.sentences())
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        text = " ".join([t["form"] for t in s["sent"]])
        encoded = self.tokenizer(text,
                                  truncation=True,
                                  padding="max_length",
                                  max_length=self.max_len,
                                  return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {"input_ids": input_ids,
                "attention_mask": attention_mask}

def output_interactions(data_raw, preds, outfile):
    with open(outfile, "w") as outf:
        for exmp, label in zip(data_raw.sentences(), preds):
            sid = exmp["sid"]
            e1 = exmp["e1"]
            e2 = exmp["e2"]
            if label != "null":
                print(sid, e1, e2, label, sep="|", file=outf)

def main():
    if len(sys.argv) != 4:
        print("Usage: predict.py model_dir datafile outfile")
        sys.exit(1)
    model_dir, datafile, outfile = sys.argv[1:4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codes = Codemaps(model_dir + ".idx")
    num_labels = codes.get_n_labels()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = BERT_CNN(model_dir, num_labels)
    head_path = os.path.join(model_dir, "head.pt")
    state = torch.load(head_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    raw = RawDataset(datafile)
    max_len = codes.maxlen
    dataset = DDIDataset(raw, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=32)
    preds = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)
    labels = [codes.idx2label(i) for i in preds]
    output_interactions(raw, labels, outfile)

if __name__ == "__main__":
    main()