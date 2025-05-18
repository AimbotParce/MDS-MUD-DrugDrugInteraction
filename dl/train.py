#!/usr/bin/env p
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


class DDIDataset(TorchDataset):
    def __init__(self, raw_dataset, codes, tokenizer, max_len):
        self.samples = list(raw_dataset.sentences())
        self.codes = codes
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.drug1_token_id = tokenizer.convert_tokens_to_ids("<DRUG1>")
        self.drug2_token_id = tokenizer.convert_tokens_to_ids("<DRUG2>")
        if (
            self.drug1_token_id == tokenizer.unk_token_id
            or self.drug2_token_id == tokenizer.unk_token_id
        ):
            tqdm.write(
                "Warning: <DRUG1> or <DRUG2> token ID not found in tokenizer. Ensure they were added.",
                file=sys.stderr,
            )

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
        input_ids = encoded["input_ids"].squeeze(0)  # Shape: (max_len)
        attention_mask = encoded["attention_mask"].squeeze(0)
        label = self.codes.label2idx(s["type"])

        # Find positions of <DRUG1> and <DRUG2> tokens
        # Fallback to CLS token (index 0) if not found, though they should be present.
        drug1_pos = (input_ids == self.drug1_token_id).nonzero(as_tuple=True)[0]
        drug2_pos = (input_ids == self.drug2_token_id).nonzero(as_tuple=True)[0]

        # Use the first occurrence if multiple are found (shouldn't happen with current data prep)
        # If a drug marker is not found, default to position 0 (CLS token) - this is a fallback.
        # A more robust solution might involve erroring or specific handling if markers are missing.
        pos_e1 = drug1_pos[0].item() if drug1_pos.nelement() > 0 else 0
        pos_e2 = drug2_pos[0].item() if drug2_pos.nelement() > 0 else 0

        if (
            pos_e1 == 0 and self.drug1_token_id != tokenizer.unk_token_id
        ):  # check if it defaulted because it wasn't found
            # This check is imperfect if drug1_token_id could legitimately be at pos 0 AND also be the CLS token ID
            # However, special tokens usually have distinct IDs.
            # tqdm.write(f"Warning: <DRUG1> not found in sample: {s['sid']}. Defaulting to pos 0.", file=sys.stderr)
            pass  # Potentially log this if it happens often
        if pos_e2 == 0 and self.drug2_token_id != tokenizer.unk_token_id:
            # tqdm.write(f"Warning: <DRUG2> not found in sample: {s['sid']}. Defaulting to pos 0.", file=sys.stderr)
            pass

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "pos_e1": torch.tensor(pos_e1, dtype=torch.long),
            "pos_e2": torch.tensor(pos_e2, dtype=torch.long),
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
        entity_head_hidden_dim=256,  # Hidden dim for MLP in entity_pool head
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
            # Classifier for concatenated entity marker embeddings
            # Each entity embedding has bert_hidden_size, so concatenated is 2 * bert_hidden_size
            self.entity_mlp = nn.Sequential(
                nn.Linear(2 * bert_hidden_size, entity_head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),  # Apply dropout within the MLP too
                nn.Linear(entity_head_hidden_dim, num_labels),
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def freeze_bert_layers(self, num_layers_to_freeze=0):
        if num_layers_to_freeze == -1:  # Freeze all of BERT
            for param in self.bert.parameters():
                param.requires_grad = False
            tqdm.write("Froze all BERT parameters.", file=sys.stderr)
            return

        if num_layers_to_freeze > 0:  # Freeze embeddings and first N encoder layers
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

            # Freeze specified number of encoder layers
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
            x = self.dropout(pooled_output)  # Apply dropout before final classification
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
            x = self.dropout(x)  # Apply dropout before final classification
            logits = self.classifier(x)
        elif self.head_type == "entity_marker":
            if pos_e1 is None or pos_e2 is None:
                raise ValueError(
                    "pos_e1 and pos_e2 must be provided for 'entity_marker' head type."
                )

            last_hidden_state = (
                outputs.last_hidden_state
            )  # (batch_size, seq_len, bert_hidden_size)

            # Gather embeddings for e1 and e2 using their positions
            # pos_e1 and pos_e2 are (batch_size), need to be (batch_size, 1, bert_hidden_size) for gather
            # or use a loop / advanced indexing
            batch_size = last_hidden_state.size(0)

            # Create index for gather. Shape: (batch_size, 1, bert_hidden_size)
            # pos_e1.unsqueeze(-1) gives (batch_size, 1)
            # .repeat(1, bert_hidden_size) would be (batch_size, bert_hidden_size)
            # We need to select one vector of size bert_hidden_size for each item in batch

            idx_e1 = (
                pos_e1.unsqueeze(1).unsqueeze(2).repeat(1, 1, bert_hidden_size)
            )  # (batch_size, 1, bert_hidden_size)
            emb_e1 = torch.gather(last_hidden_state, 1, idx_e1).squeeze(
                1
            )  # (batch_size, bert_hidden_size)

            idx_e2 = (
                pos_e2.unsqueeze(1).unsqueeze(2).repeat(1, 1, bert_hidden_size)
            )  # (batch_size, 1, bert_hidden_size)
            emb_e2 = torch.gather(last_hidden_state, 1, idx_e2).squeeze(
                1
            )  # (batch_size, bert_hidden_size)

            # Concatenate entity embeddings
            concat_emb = torch.cat(
                (emb_e1, emb_e2), dim=1
            )  # (batch_size, 2 * bert_hidden_size)
            # The self.dropout here is the general one. The MLP has its own internal dropout.
            # It might be better to remove self.dropout before self.entity_mlp if MLP handles it.
            # x = self.dropout(concat_emb) # Optional: dropout on concatenated embeddings before MLP
            logits = self.entity_mlp(concat_emb)  # MLP for classification

        return logits


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}", file=sys.stderr)

    train_raw = RawDataset(args.train_file)
    val_raw = RawDataset(args.val_file)

    max_len = args.max_len  # Use max_len from args

    bert_model_name = args.bert_model_name
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    special_tokens_dict = {
        "additional_special_tokens": ["<DRUG1>", "<DRUG2>", "<DRUG_OTHER>"]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    tqdm.write(f"Added {num_added_toks} special tokens to tokenizer.", file=sys.stderr)

    # Create codemaps *after* tokenizer is updated with special tokens if they affect vocab for other things
    # However, codemaps here is mainly for labels.
    codes = Codemaps(train_raw, max_len)
    num_labels = codes.get_n_labels()

    bert_config = AutoConfig.from_pretrained(bert_model_name)
    bert_hidden_size = bert_config.hidden_size

    cnn_kernel_sizes_list = (
        [int(k) for k in args.cnn_kernel_sizes.split(",")]
        if args.cnn_kernel_sizes
        else [3, 4, 5]
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

    # Resize token embeddings *after* model initialization and *before* loading any state dict (if applicable)
    # And *after* adding special tokens to tokenizer
    model.bert.resize_token_embeddings(len(tokenizer))
    tqdm.write(f"Resized BERT token embeddings to: {len(tokenizer)}", file=sys.stderr)
    model.to(device)

    if args.freeze_bert_layers < -1:  # Ensure valid range
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
        final_class_weights = torch.ones(
            num_labels, dtype=torch.float
        )  # Default weight 1
        for i, class_idx_in_data in enumerate(unique_labels_in_data):
            if int(class_idx_in_data) < num_labels:
                final_class_weights[int(class_idx_in_data)] = class_weights_array[i]
        class_weights = final_class_weights.to(device)
        tqdm.write(f"Calculated class weights: {class_weights}", file=sys.stderr)

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
        f"Starting training for up to {args.epochs} epochs with head_type='{args.head_type}'...",
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
            pos_e1 = batch["pos_e1"].to(device)
            pos_e2 = batch["pos_e2"].to(device)

            optimizer.zero_grad()
            if args.head_type == "entity_marker":
                logits = model(ids, mask, pos_e1=pos_e1, pos_e2=pos_e2)
            else:
                logits = model(ids, mask)  # Other heads don't need pos_e1, pos_e2

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
                pos_e1 = batch["pos_e1"].to(device)
                pos_e2 = batch["pos_e2"].to(device)

                if args.head_type == "entity_marker":
                    logits = model(ids, mask, pos_e1=pos_e1, pos_e2=pos_e2)
                else:
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
        "--bert_model_name",
        type=str,
        default="Lianglab/PharmBERT-uncased",
        help="Base BERT model name (default: Lianglab/PharmBERT-uncased)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon (default: 1e-8)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight_decay (default: 0.01)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Max epochs (default: 10)"
    )
    parser.add_argument(
        "--max_len", type=int, default=150, help="Max sequence length (default: 150)"
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate for classifier head (default: 0.5)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup steps for scheduler (default: 0)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="DataLoader num_workers (default: 0)"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (default: 3 epochs)",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Early stopping min_delta (default: 0.001)",
    )

    parser.add_argument(
        "--head_type",
        type=str,
        default="simple",
        choices=["simple", "cnn", "entity_marker"],
        help="Type of classifier head ('simple', 'cnn', 'entity_marker') (default: simple)",
    )
    parser.add_argument(
        "--cnn_out_channels",
        type=int,
        default=128,
        help="Output channels for CNN layers if head_type='cnn' (default: 128)",
    )
    parser.add_argument(
        "--cnn_kernel_sizes",
        type=str,
        default="3,4,5",
        help="Comma-separated kernel sizes for CNN layers if head_type='cnn' (default: 3,4,5)",
    )
    parser.add_argument(
        "--entity_head_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for the MLP in the 'entity_marker' head (default: 256)",
    )
    parser.add_argument(
        "--freeze_bert_layers",
        type=int,
        default=0,
        help="Number of BERT encoder layers to freeze from the bottom (embeddings also frozen if N > 0). "
        "0: BERT fully trainable. "
        "N > 0: Freeze embeddings and first N encoder layers. "
        "-1: Freeze all BERT parameters (embeddings and all encoders). (default: 0)",
    )

    args = parser.parse_args()

    if args.freeze_bert_layers < -1:
        parser.error("--freeze_bert_layers cannot be less than -1.")

    train(args)
