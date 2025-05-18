#!/usr/bin/env py
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
from tqdm import tqdm  # Import tqdm
import copy  # For deep copying model state

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
        dropout_rate=0.2,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(bert_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}", file=sys.stderr)

    train_raw = RawDataset(args.train_file)
    val_raw = RawDataset(args.val_file)

    max_len = args.max_len
    codes = Codemaps(train_raw, max_len)
    num_labels = codes.get_n_labels()

    model_name = "Lianglab/PharmBERT-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = ["<DRUG1>", "<DRUG2>", "<DRUG_OTHER>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = BERT_CNN(model_name, num_labels, dropout_rate=args.dropout_rate)
    model.to(device)
    model.bert.resize_token_embeddings(len(tokenizer))

    train_ds = DDIDataset(train_raw, codes, tokenizer, max_len)
    val_ds = DDIDataset(val_raw, codes, tokenizer, max_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_labels = [codes.label2idx(s["type"]) for s in train_raw.sentences()]
    if not train_labels:
        tqdm.write(
            "Warning: No labels found in training data. Class weights cannot be computed.",
            file=sys.stderr,
        )
        class_weights = None
    else:
        unique_labels = np.unique(train_labels)
        if (
            len(unique_labels) < 2 and num_labels > 1
        ):  # Check if only one class is present in data but more expected
            tqdm.write(
                f"Warning: Only one class ({unique_labels}) present in training data, but num_labels is {num_labels}. This might cause issues with class_weight='balanced'. Using uniform weights.",
                file=sys.stderr,
            )
            class_weights = torch.ones(num_labels, dtype=torch.float).to(device)
        else:
            class_weights_array = compute_class_weight(
                class_weight="balanced", classes=unique_labels, y=np.array(train_labels)
            )
            # Ensure class_weights_array matches num_labels (e.g. if some classes are not in train_labels)
            final_class_weights = np.ones(
                num_labels
            )  # Default to 1 for classes not in train_labels
            label_to_idx_map = {label: i for i, label in enumerate(unique_labels)}
            for i in range(num_labels):  # Iterate through all possible labels
                label_name = codes.idx2label(i)  # Get the string name of the label
                # Find if this label was in the training set and get its computed weight
                # This assumes codes.idx2label(i) and unique_labels from train_labels are comparable
                # This part is tricky if not all labels appear in train_labels.
                # For simplicity, if a label index 'i' from codes.label_index was among unique_labels from training, use its weight.
                # This requires careful mapping. A simpler approach if all labels are not guaranteed:
            if (
                len(class_weights_array) == num_labels
            ):  # Ideal case: all classes represented
                class_weights = torch.tensor(class_weights_array, dtype=torch.float).to(
                    device
                )
            else:  # If not all classes are in training data, this needs careful handling.
                # For now, we'll use the computed weights for present classes and 1.0 for absent ones.
                # This requires mapping unique_labels back to their original indices if they are not 0..N-1
                tqdm.write(
                    f"Warning: Not all {num_labels} classes were present in training data labels for weight calculation. {len(unique_labels)} found. Applying weights carefully.",
                    file=sys.stderr,
                )
                temp_weights = torch.ones(num_labels, dtype=torch.float)
                for i, class_idx in enumerate(unique_labels):
                    if int(class_idx) < num_labels:  # ensure class_idx is a valid index
                        temp_weights[int(class_idx)] = class_weights_array[i]
                class_weights = temp_weights.to(device)

            tqdm.write(f"Calculated class weights: {class_weights}", file=sys.stderr)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Early stopping parameters
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    tqdm.write(f"Starting training for up to {args.epochs} epochs...", file=sys.stderr)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} [Training]",
            unit="batch",
            leave=False,
        )
        for batch in train_pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_pbar.close()

        model.eval()
        total_val_loss, correct_val_preds = 0, 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} [Validation]",
            unit="batch",
            leave=False,
        )
        with torch.no_grad():
            for batch in val_pbar:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(ids, mask)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct_val_preds += (preds == labels).sum().item()
                val_pbar.set_postfix(val_loss=f"{loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val_preds / len(val_ds) if len(val_ds) > 0 else 0
        val_pbar.close()

        tqdm.write(
            f"Epoch {epoch+1}/{args.epochs} Summary - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}",
            file=sys.stderr,
        )

        # Early stopping check
        if avg_val_loss < best_val_loss - args.min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(
                model.state_dict()
            )  # Save the best model state
            tqdm.write(
                f"Validation loss improved to {best_val_loss:.4f}. Saving model state.",
                file=sys.stderr,
            )
        else:
            epochs_no_improve += 1
            tqdm.write(
                f"Validation loss did not improve for {epochs_no_improve} epoch(s). Best val_loss: {best_val_loss:.4f}",
                file=sys.stderr,
            )

        if epochs_no_improve >= args.patience:
            tqdm.write(
                f"Early stopping triggered after {epoch+1} epochs.", file=sys.stderr
            )
            break

    # Load the best model state before saving
    if best_model_state:
        tqdm.write(
            f"Loading best model state with val_loss: {best_val_loss:.4f}",
            file=sys.stderr,
        )
        model.load_state_dict(best_model_state)
    else:
        tqdm.write(
            "No improvement found during training or early stopping not triggered with save. Saving last model state.",
            file=sys.stderr,
        )

    # --- Saving the model ---
    out_dir = args.model_name
    os.makedirs(out_dir, exist_ok=True)

    model.bert.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # Save the state_dict of the best model (or last if no improvement)
    torch.save(model.state_dict(), os.path.join(out_dir, "head.pt"))
    codes.save(out_dir + ".idx")
    tqdm.write(
        f"Best model, tokenizer, and codemaps saved to {out_dir}", file=sys.stderr
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BERT-based model for DDI classification."
    )
    # Positional arguments
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

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW optimizer (default: 2e-5)",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer (default: 1e-8)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs (default: 10)",
    )  # Increased default
    parser.add_argument(
        "--max_len",
        type=int,
        default=150,
        help="Maximum sequence length for BERT tokenizer (default: 150)",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="Dropout rate for the classifier head (default: 0.3)",
    )  # Slightly increased default
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate scheduler (default: 0)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for DataLoader (default: 0)",
    )

    # Early stopping parameters
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Patience for early stopping (default: 3 epochs)",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Minimum delta for improvement in early stopping (default: 0.001)",
    )

    args = parser.parse_args()
    train(args)
