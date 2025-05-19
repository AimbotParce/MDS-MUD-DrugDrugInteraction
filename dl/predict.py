#!/usr/bin/env python3
import sys
import os
import argparse
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from codemaps import Codemaps
from dataset import Dataset as RawDataset
from train import BERT_CNN  # Import BERT_CNN from your train script

# Define new special marker tokens (must be identical to train.py)
E1_START_TOKEN = "[E1_START]"
E1_END_TOKEN = "[E1_END]"
E2_START_TOKEN = "[E2_START]"
E2_END_TOKEN = "[E2_END]"


class DDIPredictDataset(TorchDataset):
    def __init__(self, raw_dataset, tokenizer, max_len):
        self.samples = list(raw_dataset.sentences())
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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sid": s["sid"],
            "e1": s["e1"],
            "e2": s["e2"],
        }


def output_interactions(
    batch_outputs, outfile_handle
):  # Removed codes from here, not needed if label is string
    for item in batch_outputs:
        sid = item["sid"]
        e1 = item["e1"]
        e2 = item["e2"]
        label = item[
            "predicted_label_str"
        ]  # This is now the final label string after thresholding
        if label != "null":
            print(sid, e1, e2, label, sep="|", file=outfile_handle)


def main():
    parser = argparse.ArgumentParser(
        description="Predict DDI interactions using a trained BERT-based model with entity start/end markers."
    )
    parser.add_argument(
        "model_dir",
        help="Directory where the trained PyTorch model, tokenizer, and codemaps are saved.",
    )
    parser.add_argument("datafile", help="Path to the input data file to predict on.")
    parser.add_argument("outfile", help="Path to save the prediction results.")

    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="Lianglab/PharmBERT-uncased",
        help="Base BERT model name.",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="simple",
        choices=["simple", "cnn", "entity_marker"],
        help="Classifier head type used during training.",
    )
    parser.add_argument(
        "--cnn_out_channels",
        type=int,
        default=128,
        help="CNN output channels if head_type='cnn'.",
    )
    parser.add_argument(
        "--cnn_kernel_sizes",
        type=str,
        default="3,4,5",
        help="CNN kernel sizes if head_type='cnn'.",
    )
    parser.add_argument(
        "--entity_head_hidden_dim",
        type=int,
        default=256,
        help="MLP hidden dim for 'entity_marker' head.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for prediction."
    )

    # New argument for confidence thresholding
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.0,
        help="Minimum confidence for predicting a non-null DDI type (0.0 to 1.0). Default 0.0 means no threshold beyond argmax.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)

    codemaps_path = args.model_dir + ".idx"
    if not os.path.exists(codemaps_path):
        alt_codemaps_path = os.path.join(
            args.model_dir, os.path.basename(args.model_dir.rstrip("/\\")) + ".idx"
        )
        if os.path.exists(alt_codemaps_path):
            codemaps_path = alt_codemaps_path
        else:
            print(
                f"Error: Codemaps file not found at {args.model_dir + '.idx'} or alternative.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Loading codemaps from: {codemaps_path}", file=sys.stderr)
    codes = Codemaps(codemaps_path)
    num_labels = codes.get_n_labels()
    max_len_from_codes = codes.maxlen

    print(f"Loading tokenizer from: {args.model_dir}", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception as e:
        print(f"Error loading tokenizer from {args.model_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        bert_config = AutoConfig.from_pretrained(args.bert_model_name)
        bert_hidden_size = bert_config.hidden_size
    except Exception as e:
        print(
            f"Error loading BERT config for {args.bert_model_name}: {e}. Assuming hidden_size=768.",
            file=sys.stderr,
        )
        bert_hidden_size = 768

    cnn_kernel_sizes_list = (
        [int(k) for k in args.cnn_kernel_sizes.split(",")]
        if args.cnn_kernel_sizes
        else [3, 4, 5]
    )

    print(f"Instantiating BERT_CNN with head_type='{args.head_type}'", file=sys.stderr)
    # Note: dropout_rate is part of the model's architecture definition, but for prediction,
    # it's loaded via state_dict. If BERT_CNN needs dropout_rate in __init__ for some reason
    # other than setting the layer (which state_dict would override), this might need adjustment.
    # The current BERT_CNN uses dropout_rate to initialize nn.Dropout, so this is fine.
    # We need to pass a value for dropout_rate, e.g., the one used in training, or a default.
    # Let's add an argparse for it or use a sensible default if not critical for instantiation structure.
    # For now, assuming a default used in BERT_CNN if not critical for loading, but usually it's better
    # to instantiate with the same params. The BERT_CNN in train.py uses args.dropout_rate.
    # We should pass the dropout_rate used in training. For now, let's assume a common default if not passed.
    # Best practice: add --dropout_rate to predict.py's args.
    # For this update, I will assume BERT_CNN's dropout_rate is set during init.

    model = BERT_CNN(
        pretrained_model_name=args.bert_model_name,
        num_labels=num_labels,
        head_type=args.head_type,
        bert_hidden_size=bert_hidden_size,
        # Pass a default dropout_rate; the actual trained dropout layer's state (if any)
        # or just the probability (for nn.Dropout) is part of the model class,
        # and its effect during eval() mode is typically no dropout.
        # The key is that the architecture matches for state_dict loading.
        dropout_rate=0.3,  # Example default, or add as arg. Value during eval() is what matters.
        cnn_out_channels=args.cnn_out_channels,
        cnn_kernel_sizes=cnn_kernel_sizes_list,
        entity_head_hidden_dim=args.entity_head_hidden_dim,
    )

    model.bert.resize_token_embeddings(len(tokenizer))
    print(f"Resized BERT token embeddings to: {len(tokenizer)}", file=sys.stderr)

    model_state_path = os.path.join(args.model_dir, "model_state_dict.pt")
    print(f"Loading model state from: {model_state_path}", file=sys.stderr)
    try:
        state_dict = torch.load(model_state_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(
            f"Error loading model state_dict from {model_state_path}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    model.to(device)
    model.eval()

    print(f"Loading and preparing dataset from: {args.datafile}", file=sys.stderr)
    raw_dataset_to_predict = RawDataset(args.datafile)
    predict_dataset = DDIPredictDataset(
        raw_dataset_to_predict, tokenizer, max_len_from_codes
    )
    predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size)

    print(
        f"Starting prediction with confidence threshold: {args.confidence_threshold}...",
        file=sys.stderr,
    )
    all_predictions_for_output = []

    with torch.no_grad(), open(args.outfile, "w") as outfile_handle:
        for batch in tqdm(predict_loader, desc="Predicting", unit="batch"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            logits = model(ids, mask)
            probs = torch.softmax(logits, dim=1)

            batch_max_probs, batch_preds_indices = torch.max(probs, dim=1)

            batch_preds_indices_cpu = batch_preds_indices.cpu().tolist()
            batch_max_probs_cpu = batch_max_probs.cpu().tolist()

            for i, pred_idx in enumerate(batch_preds_indices_cpu):
                predicted_label_str_raw = codes.idx2label(pred_idx)
                confidence = batch_max_probs_cpu[i]

                final_label_to_output = predicted_label_str_raw
                # Apply threshold only if the predicted class is not "null"
                if (
                    predicted_label_str_raw != "null"
                    and confidence < args.confidence_threshold
                ):
                    final_label_to_output = "null"

                all_predictions_for_output.append(
                    {
                        "sid": batch["sid"][i],
                        "e1": batch["e1"][i],
                        "e2": batch["e2"][i],
                        "predicted_label_str": final_label_to_output,
                    }
                )

        output_interactions(all_predictions_for_output, outfile_handle)

    print(f"Predictions saved to {args.outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
