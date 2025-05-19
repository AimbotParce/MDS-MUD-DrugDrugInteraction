#!/usr/bin/
#!/usr/bin/env python3
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import copy

from codemaps import Codemaps
from dataset import Dataset as RawDataset


E1_START_TOKEN = "[E1_START]"
E1_END_TOKEN = "[E1_END]"
E2_START_TOKEN = "[E2_START]"
E2_END_TOKEN = "[E2_END]"
NEW_SPECIAL_TOKENS = [E1_START_TOKEN, E1_END_TOKEN, E2_START_TOKEN, E2_END_TOKEN]


class DDIDataset(TorchDataset):
    def __init__(self, raw_dataset, codes, tokenizer, max_len):
        self.samples = list(raw_dataset.sentences())
        self.codes = codes
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def _insert_markers(self, token_list):
        new_token_list = []
        for token_dict in token_list:
            form = token_dict["form"]
            if form == "<DRUG1>":
                new_token_list.append({"form": E1_START_TOKEN})
                new_token_list.append(token_dict)
                new_token_list.append({"form": E1_END_TOKEN})
            elif form == "<DRUG2>":
                new_token_list.append({"form": E2_START_TOKEN})
                new_token_list.append(token_dict)
                new_token_list.append({"form": E2_END_TOKEN})
            else:
                new_token_list.append(token_dict)
        return new_token_list

    def __getitem__(self, idx):
        s = self.samples[idx]

        marked_token_list = self._insert_markers(s["sent"])
        text_to_tokenize = " ".join([t["form"] for t in marked_token_list])

        encoded = self.tokenizer(
            text_to_tokenize,
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
        dropout_rate=0.3,
        head_type="simple",
        bert_hidden_size=768,
        cnn_out_channels=128,
        cnn_kernel_sizes=[3, 4, 5],
        entity_head_hidden_dim=256,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.head_type = head_type
        self.dropout = nn.Dropout(dropout_rate)

        if self.head_type == "simple":
            self.classifier = nn.Linear(bert_hidden_size, num_labels)
        elif self.head_type == "cnn":
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=bert_hidden_size,
                        out_channels=cnn_out_channels,
                        kernel_size=k,
                        padding=(k - 1) // 2,
                    )
                    for k in cnn_kernel_sizes
                ]
            )
            self.classifier = nn.Linear(
                len(cnn_kernel_sizes) * cnn_out_channels, num_labels
            )
        elif self.head_type == "entity_marker":
            self.entity_mlp = nn.Sequential(
                nn.Linear(2 * bert_hidden_size, entity_head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(entity_head_hidden_dim, num_labels),
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def freeze_bert_layers(self, num_layers_to_freeze=0):
        if num_layers_to_freeze == -1:
            for param in self.bert.parameters():
                param.requires_grad = False
            tqdm.write("Froze all BERT parameters.", file=sys.stderr)
            return

        if num_layers_to_freeze > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

            for layer_idx in range(
                min(num_layers_to_freeze, len(self.bert.encoder.layer))
            ):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False
            tqdm.write(
                f"Froze BERT embeddings and the first {min(num_layers_to_freeze, len(self.bert.encoder.layer))} encoder layers.",
                file=sys.stderr,
            )
        elif num_layers_to_freeze == 0:
            tqdm.write(
                "BERT model is fully trainable (no layers frozen by this method).",
                file=sys.stderr,
            )

    def forward(self, input_ids, attention_mask, pos_e1=None, pos_e2=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.head_type == "simple":
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.classifier(x)
        elif self.head_type == "cnn":
            last_hidden_state = outputs.last_hidden_state
            x = last_hidden_state.permute(0, 2, 1)
            conv_outputs = []
            for conv_layer in self.convs:
                conv_out = F.relu(conv_layer(x))
                pooled_out = F.max_pool1d(
                    conv_out, kernel_size=conv_out.size(2)
                ).squeeze(2)
                conv_outputs.append(pooled_out)
            x = torch.cat(conv_outputs, dim=1)
            x = self.dropout(x)
            logits = self.classifier(x)
        elif self.head_type == "entity_marker":
            if pos_e1 is None or pos_e2 is None:
                raise ValueError(
                    "pos_e1 and pos_e2 must be provided for 'entity_marker' head type if it were used."
                )
            last_hidden_state = outputs.last_hidden_state
            bert_hidden_size = last_hidden_state.size(2)  # Get hidden_size dynamically
            idx_e1 = (
                pos_e1.unsqueeze(1).unsqueeze(2).expand(-1, -1, bert_hidden_size)
            )  # Use expand
            emb_e1 = torch.gather(last_hidden_state, 1, idx_e1).squeeze(1)
            idx_e2 = (
                pos_e2.unsqueeze(1).unsqueeze(2).expand(-1, -1, bert_hidden_size)
            )  # Use expand
            emb_e2 = torch.gather(last_hidden_state, 1, idx_e2).squeeze(1)
            concat_emb = torch.cat((emb_e1, emb_e2), dim=1)
            logits = self.entity_mlp(concat_emb)
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type} in forward pass")

        return logits


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}", file=sys.stderr)

    train_raw = RawDataset(args.train_file)
    val_raw = RawDataset(args.val_file)

    max_len = args.max_len

    bert_model_name = args.bert_model_name
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    num_added_toks = tokenizer.add_special_tokens(
        {"additional_special_tokens": NEW_SPECIAL_TOKENS}
    )
    tqdm.write(
        f"Added {num_added_toks} new special tokens to tokenizer: {NEW_SPECIAL_TOKENS}",
        file=sys.stderr,
    )

    codes = Codemaps(train_raw, max_len)
    num_labels = codes.get_n_labels()

    bert_config = AutoConfig.from_pretrained(bert_model_name)
    bert_hidden_size = bert_config.hidden_size

    cnn_kernel_sizes_list = (
        [int(k) for k in args.cnn_kernel_sizes.split(",")]
        if args.cnn_kernel_sizes
        else [3, 4, 5]
    )

    if args.head_type != "simple":
        tqdm.write(
            f"Warning: Current data processing (entity start/end markers) is primarily designed for 'simple' head_type. "
            f"You are using '{args.head_type}'. Ensure this is intended.",
            file=sys.stderr,
        )

    model = BERT_CNN(
        pretrained_model_name=bert_model_name,
        num_labels=num_labels,
        dropout_rate=args.dropout_rate,
        head_type=args.head_type,
        bert_hidden_size=bert_hidden_size,
        cnn_out_channels=args.cnn_out_channels,
        cnn_kernel_sizes=cnn_kernel_sizes_list,
        entity_head_hidden_dim=args.entity_head_hidden_dim,
    )

    model.bert.resize_token_embeddings(len(tokenizer))
    tqdm.write(f"Resized BERT token embeddings to: {len(tokenizer)}", file=sys.stderr)
    model.to(device)

    if args.freeze_bert_layers < -1:
        raise ValueError("--freeze_bert_layers cannot be less than -1.")
    model.freeze_bert_layers(num_layers_to_freeze=args.freeze_bert_layers)

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
        unique_labels_in_data = np.unique(train_labels)
        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=unique_labels_in_data,
            y=np.array(train_labels),
        )
        final_class_weights_tensor = torch.ones(num_labels, dtype=torch.float)
        for i, class_idx_in_data in enumerate(unique_labels_in_data):
            if int(class_idx_in_data) < num_labels:
                final_class_weights_tensor[int(class_idx_in_data)] = (
                    class_weights_array[i]
                )

        # Apply weight capping if max_weight is specified and > 0
        if args.max_class_weight > 0:
            final_class_weights_tensor = torch.clamp(
                final_class_weights_tensor, max=args.max_class_weight
            )
            tqdm.write(
                f"Applied class weight capping. Max weight is now {args.max_class_weight}.",
                file=sys.stderr,
            )

        class_weights = final_class_weights_tensor.to(device)
        tqdm.write(f"Final class weights: {class_weights}", file=sys.stderr)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    tqdm.write(
        f"Starting training for up to {args.epochs} epochs with input strategy: entity start/end markers and head_type='{args.head_type}'...",
        file=sys.stderr,
    )
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
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                args.max_grad_norm,
            )
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

                val_batch_loss = criterion(logits, labels)
                total_val_loss += val_batch_loss.item()

                preds = torch.argmax(logits, dim=1)
                correct_val_preds += (preds == labels).sum().item()
                val_pbar.set_postfix(val_loss=f"{val_batch_loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val_preds / len(val_ds) if len(val_ds) > 0 else 0
        val_pbar.close()

        tqdm.write(
            f"Epoch {epoch+1}/{args.epochs} Summary - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}",
            file=sys.stderr,
        )

        if avg_val_loss < best_val_loss - args.min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
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

    if best_model_state:
        tqdm.write(
            f"Loading best model state with val_loss: {best_val_loss:.4f}",
            file=sys.stderr,
        )
        model.load_state_dict(best_model_state)
    else:
        tqdm.write(
            "No improvement found or early stopping not triggered with save. Saving last model state.",
            file=sys.stderr,
        )

    out_dir = args.model_name
    os.makedirs(out_dir, exist_ok=True)

    tokenizer.save_pretrained(out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
    codes.save(out_dir + ".idx")
    tqdm.write(
        f"Best model state dict, tokenizer, and codemaps saved to {out_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BERT-based model for DDI classification with entity start/end markers."
    )
    parser.add_argument("train_file", help="Path to the training data file")
    parser.add_argument("val_file", help="Path to the validation data file")
    parser.add_argument("model_name", help="Directory name to save the trained model")

    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="Lianglab/PharmBERT-uncased",
        help="Base BERT model name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size"
    )  # Defaulting to your last successful batch size
    parser.add_argument(
        "--lr", type=float, default=1.5e-5, help="Learning rate"
    )  # Defaulting to your last successful LR
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.03, help="AdamW weight_decay"
    )  # Defaulting to your last successful WD
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs")
    parser.add_argument(
        "--max_len",
        type=int,
        default=150,
        help="Max sequence length for BERT tokenizer",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate for classifier head",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Warmup steps for scheduler"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="DataLoader num_workers"
    )

    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0.001, help="Early stopping min_delta"
    )

    parser.add_argument(
        "--head_type",
        type=str,
        default="simple",
        choices=["simple", "cnn", "entity_marker"],
        help="Classifier head type. 'simple' is recommended for entity start/end marker strategy.",
    )

    parser.add_argument(
        "--cnn_out_channels",
        type=int,
        default=128,
        help="CNN output channels if head_type='cnn'",
    )
    parser.add_argument(
        "--cnn_kernel_sizes",
        type=str,
        default="3,4,5",
        help="CNN kernel sizes if head_type='cnn'",
    )
    parser.add_argument(
        "--entity_head_hidden_dim",
        type=int,
        default=256,
        help="MLP hidden dim for 'entity_marker' head",
    )

    parser.add_argument(
        "--freeze_bert_layers",
        type=int,
        default=6,  # Defaulting to your last successful freeze setting
        help="Number of BERT encoder layers to freeze (embeddings also frozen if N > 0). "
        "0: Fully trainable. N > 0: Freeze embeddings and first N encoder layers. -1: Freeze all BERT.",
    )

    # New argument for class weight capping
    parser.add_argument(
        "--max_class_weight",
        type=float,
        default=0,  # Default 0 means no capping
        help="Maximum value to cap class weights at. Set to 0 for no capping (default: 0). Try values like 10 or 15.",
    )

    args = parser.parse_args()

    if args.freeze_bert_layers < -1:
        parser.error("--freeze_bert_layers cannot be less than -1.")
    if args.head_type != "simple":
        print(
            f"Warning: You are using head_type='{args.head_type}'. "
            "The current data processing strategy (entity start/end markers) "
            "is primarily designed for head_type='simple'.",
            file=sys.stderr,
        )

    train(args)
