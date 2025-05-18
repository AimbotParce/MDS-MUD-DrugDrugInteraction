#!/usr/bin/env python3
import sys
import os
import argparse  # For command-line arguments
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer, AutoConfig  # Added AutoConfig

from codemaps import Codemaps
from dataset import Dataset as RawDataset

# Ensure train.py is in the same directory or Python path to import BERT_CNN
from train import BERT_CNN  # Assuming train.py contains the BERT_CNN class definition


class DDIPredictDataset(
    TorchDataset
):  # Renamed to avoid conflict if train.py also has DDIDataset
    def __init__(self, raw_dataset, tokenizer, max_len):
        self.samples = list(raw_dataset.sentences())
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = " ".join([t["form"] for t in s["sent"]])
        # Ensure consistent tokenization with training
        # The special tokens <DRUG1>, <DRUG2>, <DRUG_OTHER> should be handled by the tokenizer
        # if they were added during its saving process in train.py
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # We don't have labels during prediction, but return a structure
        # consistent with what the model might expect if it were processing batches with labels
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sid": s["sid"],  # Keep sid, e1, e2 for output generation
            "e1": s["e1"],
            "e2": s["e2"],
        }


def output_interactions(batch_outputs, codes, outfile_handle):
    """
    Writes interactions from a batch of predictions to the output file.
    batch_outputs should be a list of dictionaries, where each dict contains
    'sid', 'e1', 'e2', and 'predicted_label_str'.
    """
    for item in batch_outputs:
        sid = item["sid"]
        e1 = item["e1"]
        e2 = item["e2"]
        label = item["predicted_label_str"]
        if label != "null":  # Only output if it's an interaction
            print(sid, e1, e2, label, sep="|", file=outfile_handle)


def main():
    parser = argparse.ArgumentParser(
        description="Predict DDI interactions using a trained BERT-based model."
    )
    parser.add_argument(
        "model_dir",
        help="Directory where the trained PyTorch model, tokenizer, and codemaps are saved.",
    )
    parser.add_argument(
        "datafile",
        help="Path to the input data file to predict on (parsed .pck or XML directory).",
    )
    parser.add_argument("outfile", help="Path to save the prediction results.")

    # Arguments to match model architecture used in training
    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="Lianglab/PharmBERT-uncased",
        help="Base BERT model name used for initialization (e.g., 'Lianglab/PharmBERT-uncased'). Must match training.",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="simple",
        choices=["simple", "cnn"],
        help="Type of classifier head used during training ('simple' or 'cnn').",
    )
    parser.add_argument(
        "--cnn_out_channels",
        type=int,
        default=128,
        help="Output channels for CNN layers if head_type='cnn'.",
    )
    parser.add_argument(
        "--cnn_kernel_sizes",
        type=str,
        default="3,4,5",
        help="Comma-separated kernel sizes for CNN layers if head_type='cnn' (e.g., '3,4,5').",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for prediction."
    )
    # dropout_rate is part of the model architecture, so it's implicitly handled by loading the state_dict

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)

    # 1. Load Codemaps
    # The .idx file is saved directly with model_name as prefix by train.py
    codemaps_path = args.model_dir + ".idx"
    if not os.path.exists(codemaps_path):
        # Fallback: try looking for .idx inside model_dir if model_dir was given without .idx suffix
        # e.g. if model_name in train was "my_model" and model_dir here is "my_model"
        # then codes.save("my_model.idx")
        # if model_name in train was "output_models/my_model", then codes.save("output_models/my_model.idx")
        # This heuristic might be needed if the user provides just the directory.
        # For now, assume model_dir is the prefix used for .idx
        alt_codemaps_path = os.path.join(
            args.model_dir, os.path.basename(args.model_dir) + ".idx"
        )
        if os.path.exists(alt_codemaps_path):
            codemaps_path = alt_codemaps_path
        else:  # Try common pattern if model_dir is a directory and idx file is inside with a fixed name like 'codes.idx'
            fixed_name_codemaps_path = os.path.join(
                args.model_dir, "codes.idx"
            )  # Or whatever train.py might save
            # The train.py saves it as `out_dir + ".idx"`. So if out_dir is "my_model_directory", it's "my_model_directory.idx"
            # This means the user should pass "my_model_directory" as model_dir, and we append ".idx"
            # The current train.py saves codes as: codes.save(out_dir + ".idx")
            # So if out_dir = "ddi_bert_cnn_model", it saves "ddi_bert_cnn_model.idx"
            # The predict script's model_dir should be "ddi_bert_cnn_model"
            pass  # codemaps_path is already model_dir + ".idx" which is correct.

    print(f"Loading codemaps from: {codemaps_path}", file=sys.stderr)
    codes = Codemaps(codemaps_path)  # Loads from file because maxlen is None
    num_labels = codes.get_n_labels()
    max_len_from_codes = codes.maxlen  # Get max_len from saved codemaps

    # 2. Load Tokenizer
    print(f"Loading tokenizer from: {args.model_dir}", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception as e:
        print(f"Error loading tokenizer from {args.model_dir}: {e}", file=sys.stderr)
        print(
            "Ensure that the 'model_dir' contains the tokenizer files (vocab.txt, tokenizer_config.json, etc.) "
            "saved by train.py, and that it's not a Keras model path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Instantiate Model
    # Get bert_hidden_size from the config of the base BERT model
    try:
        bert_config = AutoConfig.from_pretrained(args.bert_model_name)
        bert_hidden_size = bert_config.hidden_size
    except Exception as e:
        print(
            f"Error loading BERT config for {args.bert_model_name} to get hidden_size: {e}",
            file=sys.stderr,
        )
        # Fallback or error
        bert_hidden_size = 768  # Default for BERT-base, but risky
        print(
            f"Warning: Could not determine bert_hidden_size automatically. Assuming {bert_hidden_size}.",
            file=sys.stderr,
        )

    cnn_kernel_sizes_list = (
        [int(k) for k in args.cnn_kernel_sizes.split(",")]
        if args.cnn_kernel_sizes
        else [3, 4, 5]
    )

    print(f"Instantiating BERT_CNN with head_type='{args.head_type}'", file=sys.stderr)
    model = BERT_CNN(
        pretrained_model_name=args.bert_model_name,  # Base model for structure
        num_labels=num_labels,
        # dropout_rate is part of the saved state_dict, so not needed here for instantiation if loaded correctly
        head_type=args.head_type,
        bert_hidden_size=bert_hidden_size,
        cnn_out_channels=args.cnn_out_channels,
        cnn_kernel_sizes=cnn_kernel_sizes_list,
    )

    # 4. Load Saved State Dictionary
    # The train.py saves it as "model_state_dict.pt" inside the model_dir
    model_state_path = os.path.join(args.model_dir, "model_state_dict.pt")
    print(f"Loading model state from: {model_state_path}", file=sys.stderr)
    try:
        state_dict = torch.load(model_state_path, map_location=device)
        model.load_state_dict(state_dict)  # strict=True by default
    except Exception as e:
        print(
            f"Error loading model state_dict from {model_state_path}: {e}",
            file=sys.stderr,
        )
        print(
            "Ensure the path is correct and the model architecture in predict.py matches the saved model.",
            file=sys.stderr,
        )
        sys.exit(1)

    model.to(device)
    model.eval()

    # 5. Prepare Dataset and DataLoader
    print(f"Loading and preparing dataset from: {args.datafile}", file=sys.stderr)
    raw_dataset_to_predict = RawDataset(args.datafile)
    # Use max_len from the codemaps used during training
    predict_dataset = DDIPredictDataset(
        raw_dataset_to_predict, tokenizer, max_len_from_codes
    )
    predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size)

    # 6. Prediction Loop
    print("Starting prediction...", file=sys.stderr)
    all_predictions_for_output = []  # To store dicts for output_interactions

    with torch.no_grad(), open(args.outfile, "w") as outfile_handle:
        for batch in tqdm(predict_loader, desc="Predicting", unit="batch"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            logits = model(ids, mask)
            batch_preds_indices = torch.argmax(logits, dim=1).cpu().tolist()

            # Store predictions with their original info for output
            for i, pred_idx in enumerate(batch_preds_indices):
                all_predictions_for_output.append(
                    {
                        "sid": batch["sid"][
                            i
                        ],  # Assuming DDIPredictDataset returns these
                        "e1": batch["e1"][i],
                        "e2": batch["e2"][i],
                        "predicted_label_str": codes.idx2label(pred_idx),
                    }
                )

        # Write all collected predictions to file
        output_interactions(all_predictions_for_output, codes, outfile_handle)

    print(f"Predictions saved to {args.outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
