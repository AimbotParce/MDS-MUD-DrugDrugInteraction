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
        dropout_rate=0.3,
        head_type="simple",  # 'simple' or 'cnn'
        bert_hidden_size=768,  # Default for BERT base models
        cnn_out_channels=128,  # Output channels for CNN layers
        cnn_kernel_sizes=[3, 4, 5],  # Kernel sizes for CNN layers (example)
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.head_type = head_type
        self.dropout = nn.Dropout(dropout_rate)

        if self.head_type == "simple":
            self.classifier = nn.Linear(bert_hidden_size, num_labels)
        elif self.head_type == "cnn":
            # CNN layers
            # Input to Conv1d: (batch_size, in_channels, seq_len)
            # BERT last_hidden_state: (batch_size, seq_len, bert_hidden_size)
            # We permute it to (batch_size, bert_hidden_size, seq_len)
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=bert_hidden_size,
                        out_channels=cnn_out_channels,
                        kernel_size=k,
                        padding=(k - 1) // 2,
                    )  # 'same' padding
                    for k in cnn_kernel_sizes
                ]
            )
            # The output size after convs and pooling depends on the number of kernels and their out_channels
            # If using GlobalMaxPooling, each conv layer output (after pooling) will be (batch_size, cnn_out_channels)
            # Concatenating them will result in (batch_size, len(cnn_kernel_sizes) * cnn_out_channels)
            # For Flatten, it's more complex. Let's use GlobalMaxPooling for simplicity and robustness here.
            # self.flatten = nn.Flatten() # Alternative to Global Max Pooling

            # Classifier for CNN head
            # Each Conv1D will produce cnn_out_channels. If we concatenate features from multiple kernel sizes:
            self.classifier = nn.Linear(
                len(cnn_kernel_sizes) * cnn_out_channels, num_labels
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def freeze_bert_layers(self, num_layers_to_freeze=None):
        """
        Freezes BERT layers.
        If num_layers_to_freeze is None, freezes all BERT parameters.
        If num_layers_to_freeze is an int, freezes the embedding layer and the first
        `num_layers_to_freeze` encoder layers.
        """
        if num_layers_to_freeze is None:  # Freeze all of BERT
            for param in self.bert.parameters():
                param.requires_grad = False
            tqdm.write("Froze all BERT parameters.", file=sys.stderr)
            return

        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of encoder layers
        if num_layers_to_freeze > 0:
            for layer_idx in range(
                min(num_layers_to_freeze, len(self.bert.encoder.layer))
            ):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False
            tqdm.write(
                f"Froze BERT embeddings and the first {min(num_layers_to_freeze, len(self.bert.encoder.layer))} encoder layers.",
                file=sys.stderr,
            )
        else:
            tqdm.write(
                "Froze BERT embeddings. All encoder layers are trainable.",
                file=sys.stderr,
            )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.head_type == "simple":
            # Use pooler_output for simple classification
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.classifier(x)
        elif self.head_type == "cnn":
            # Use last_hidden_state for CNN head
            last_hidden_state = (
                outputs.last_hidden_state
            )  # (batch_size, seq_len, bert_hidden_size)

            # Permute to (batch_size, bert_hidden_size, seq_len) for Conv1D
            x = last_hidden_state.permute(0, 2, 1)

            # Apply Conv layers, ReLU, and Global Max Pooling
            conv_outputs = []
            for conv_layer in self.convs:
                conv_out = F.relu(
                    conv_layer(x)
                )  # (batch_size, cnn_out_channels, seq_len)
                # Global Max Pooling over the sequence length dimension
                pooled_out = F.max_pool1d(
                    conv_out, kernel_size=conv_out.size(2)
                ).squeeze(
                    2
                )  # (batch_size, cnn_out_channels)
                conv_outputs.append(pooled_out)

            # Concatenate features from different kernel sizes
            x = torch.cat(
                conv_outputs, dim=1
            )  # (batch_size, len(cnn_kernel_sizes) * cnn_out_channels)
            x = self.dropout(x)
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

    # Determine BERT model name and hidden size
    # Using PharmBERT as a good default, can be made an argument
    bert_model_name = "Lianglab/PharmBERT-uncased"
    # Fetch config to get hidden_size, or assume 768 for base models
    from transformers import AutoConfig

    bert_config = AutoConfig.from_pretrained(bert_model_name)
    bert_hidden_size = bert_config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    special_tokens = ["<DRUG1>", "<DRUG2>", "<DRUG_OTHER>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Parse CNN kernel sizes from argument
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
    )
    model.to(device)
    model.bert.resize_token_embeddings(len(tokenizer))

    # Apply layer freezing if specified
    if args.freeze_bert_layers is not None:
        if args.freeze_bert_layers == -1:  # Special value to freeze all BERT
            model.freeze_bert_layers(num_layers_to_freeze=None)
        elif args.freeze_bert_layers >= 0:
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
        unique_labels = np.unique(train_labels)
        # Ensure all labels defined in codes are considered for weights array
        all_possible_labels_indices = np.arange(num_labels)

        # Compute weights only for classes present in the training data
        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=unique_labels,  # Only provide unique labels found in data
            y=np.array(train_labels),
        )

        # Map these weights to the full list of labels
        # Initialize weights for all labels to 1.0 (or some other default)
        final_class_weights = torch.ones(num_labels, dtype=torch.float)
        for i, class_idx_in_data in enumerate(unique_labels):
            # class_idx_in_data is the actual label index (e.g., 0, 1, 3 if label 2 is missing)
            if int(class_idx_in_data) < num_labels:  # Ensure it's a valid index
                final_class_weights[int(class_idx_in_data)] = class_weights_array[i]

        class_weights = final_class_weights.to(device)
        tqdm.write(f"Calculated class weights: {class_weights}", file=sys.stderr)

    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad, model.parameters()
        ),  # Only pass trainable parameters
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

    # Save only the BERT part if it was fine-tuned, or the whole model if head is custom
    # For simplicity and to ensure the head is saved, save the whole model's state_dict.
    # The AutoModel.save_pretrained approach is more for sharing/reloading the BERT part itself.

    # To save the fine-tuned BERT base model separately (optional):
    # model.bert.save_pretrained(os.path.join(out_dir, "bert_fine_tuned"))

    tokenizer.save_pretrained(out_dir)
    torch.save(
        model.state_dict(), os.path.join(out_dir, "model_state_dict.pt")
    )  # Changed filename for clarity
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
        "--batch_size", type=int, default=128, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate (default: 2e-5)"
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
        default=0.3,
        help="Dropout rate for classifier head (default: 0.3)",
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

    # New arguments for model architecture
    parser.add_argument(
        "--head_type",
        type=str,
        default="simple",
        choices=["simple", "cnn"],
        help="Type of classifier head to use on top of BERT ('simple' or 'cnn') (default: simple)",
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
        help="Comma-separated kernel sizes for CNN layers if head_type='cnn' (e.g., '3,4,5') (default: 3,4,5)",
    )
    parser.add_argument(
        "--freeze_bert_layers",
        type=int,
        default=0,
        help="Number of BERT encoder layers to freeze from the bottom (embeddings always frozen if > -1). "
        "-1 freezes all of BERT. 0 means only embeddings are frozen if specified, "
        "but all encoder layers are trainable. (default: 0, meaning no encoder layers frozen by default beyond embeddings if any). "
        "To make all BERT trainable (including embeddings), this should not be set or a different logic is needed. "
        "Current logic: 0 means embeddings frozen, encoders trainable. >0 means embeddings + N encoders frozen. -1 means all BERT frozen."
        "Let's adjust: 0 means nothing frozen. N > 0 freezes embeddings + N-1 encoders. -1 freezes all. No, this is confusing."
        "New logic: 0 = nothing frozen. N > 0 = freeze embeddings and first N encoder layers. -1 = freeze all BERT parameters (embeddings + all encoders)."
        "Corrected logic: freeze_bert_layers=0 means BERT is fully trainable. freeze_bert_layers=N (N>0) freezes embeddings and first N encoder layers. freeze_bert_layers=-1 freezes all BERT parameters."
        "Revisiting freeze_bert_layers argument: 0 = no layers frozen. N > 0 = freeze embeddings and first N encoder layers. -1 = freeze all BERT params."
        "Final decision on freeze_bert_layers: 0 = BERT fully trainable. N > 0 = freeze embeddings and first N *encoder* layers. -1 = freeze all BERT parameters (embeddings and all encoder layers)."
        "Actually, a simpler scheme: 0 = fully trainable. N > 0 freezes the first N encoder layers *and* embeddings. -1 freezes all parameters of BERT model. For this implementation, let's use: N>=0: freeze embeddings and first N encoder layers. If N=-1, freeze all of BERT. If N is not specified (default, e.g. 0 in argparse), don't freeze beyond what the pre-trained model specifies."
        "Let's make the default for freeze_bert_layers 0, meaning no layers are frozen by default. If the user specifies a positive number N, we freeze embeddings and the first N encoder layers. If -1, freeze all of BERT.",
    )

    args = parser.parse_args()

    # Refined logic for freeze_bert_layers based on its new default and meaning
    if args.freeze_bert_layers < -1:
        parser.error("--freeze_bert_layers cannot be less than -1.")

    train(args)
