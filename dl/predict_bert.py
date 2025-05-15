import argparse
import json
import os
from xml.dom import minidom
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import numpy as np


def parse_ddi_data_for_prediction(data_input_path):
    """
    Parse DDI XML files from a directory or a single XML file for prediction.
    Returns a list of dictionaries, each containing:
    'sid': sentence ID
    'e1_id': ID of the first entity in the pair
    'e2_id': ID of the second entity in the pair
    'text': sentence text with entities marked (e.g., "[E1] text [/E1]")
    """
    parsed_examples = []
    files_to_process = []

    if os.path.isdir(data_input_path):
        for fname in sorted(os.listdir(data_input_path)):
            if fname.lower().endswith(".xml"):
                files_to_process.append(os.path.join(data_input_path, fname))
    elif os.path.isfile(data_input_path) and data_input_path.lower().endswith(".xml"):
        files_to_process.append(data_input_path)
    else:
        print(
            f"Error: Input path '{data_input_path}' is not a valid directory or .xml file."
        )
        return parsed_examples

    for filepath in files_to_process:
        try:
            doc = minidom.parse(filepath)
            sentences = doc.getElementsByTagName("sentence")
            for s_node in sentences:
                s_id = s_node.getAttribute("id")
                stext_original = s_node.getAttribute("text")

                entities_in_sentence = {
                    e.getAttribute("id"): e
                    for e in s_node.getElementsByTagName("entity")
                }

                pairs = s_node.getElementsByTagName("pair")
                for p_node in pairs:
                    e1_attr_id = p_node.getAttribute("e1")
                    e2_attr_id = p_node.getAttribute("e2")

                    ent1_node = entities_in_sentence.get(e1_attr_id)
                    ent2_node = entities_in_sentence.get(e2_attr_id)

                    if not ent1_node or not ent2_node:
                        print(
                            f"Warning: Could not find entity nodes for pair {e1_attr_id}-{e2_attr_id} in sentence {s_id}. Skipping pair."
                        )
                        continue

                    ent1_text = ent1_node.getAttribute("text")
                    ent2_text = ent2_node.getAttribute("text")

                    # IMPORTANT: This marking strategy must be identical to the one used during training.
                    # Training script used:
                    # marked = stext.replace(ent1_text, f"[E1] {ent1_text} [/E1]", 1).replace(
                    #    ent2_text, f"[E2] {ent2_text} [/E2]", 1
                    # )
                    # This means E1 is marked first, then E2 is marked in the modified string.

                    marked_text = stext_original

                    # Create the marked versions of entity texts
                    e1_marker_open = "[E1]"
                    e1_marker_close = "[/E1]"
                    e2_marker_open = "[E2]"
                    e2_marker_close = "[/E2]"

                    # To handle cases where entity texts might be identical or one contains the other,
                    # we need to be careful. The training script's simple .replace(..., 1) sequence
                    # implies a specific order and behavior.

                    # Step 1: Replace first entity e1
                    # Find the first occurrence of ent1_text and replace it
                    pos_e1 = marked_text.find(ent1_text)
                    if pos_e1 != -1:
                        marked_text = (
                            marked_text[:pos_e1]
                            + f"{e1_marker_open} {ent1_text} {e1_marker_close}"
                            + marked_text[pos_e1 + len(ent1_text) :]
                        )
                    else:
                        # This case should ideally not happen if XML is well-formed and entities are in sentence text
                        print(
                            f"Warning: Entity text '{ent1_text}' not found in sentence for E1 marking. Sentence: {stext_original}"
                        )

                    # Step 2: Replace second entity e2 in the already modified string
                    # Find the first occurrence of ent2_text (that hasn't been incorporated into an E1 marker complex)
                    # This is the tricky part of replicating the original .replace().replace()
                    # If ent1_text and ent2_text are different, this is usually fine.
                    # If ent1_text and ent2_text are the same, the second replace will target the *next* occurrence.

                    # A more robust way to handle the second replacement if texts might be identical or overlap
                    # with the first replacement's markers is complex.
                    # Given the training script's approach: `string.replace(e1).replace(e2)`
                    # we replicate that sequential replacement.

                    pos_e2 = marked_text.find(
                        ent2_text
                    )  # Find in the string possibly already modified by E1

                    # We need to ensure we are not replacing ent2_text if it's now part of the E1 markers
                    # For example, if ent1_text = "drug" and ent2_text = "drug"
                    # stext = "drug A and drug B"
                    # After E1: "[E1] drug [/E1] A and drug B"
                    # Now find "drug" for E2. The first "drug" is inside E1 markers.
                    # The .replace(ent2_text, ..., 1) in python will find the first occurrence of ent2_text
                    # that is not already part of a more complex string that was introduced.

                    # Let's assume the original simple sequential replace:
                    if pos_e2 != -1:
                        # Check if this ent2_text is actually the E1 text we just marked, if texts are same
                        # This check is to prevent re-marking the same physical text span if e1_text == e2_text
                        # and they refer to the same text span (though XML structure implies distinct entities e1, e2)

                        # The training script's replace(e1,...).replace(e2,...) implicitly handles this:
                        # if e1_text == e2_text, first replace hits first occurrence for e1.
                        # second replace hits *next* occurrence for e2.

                        # Simplified: assume the training's .replace().replace() worked by finding
                        # the first available raw entity text for each replacement.
                        current_search_offset = 0
                        while current_search_offset < len(marked_text):
                            pos_e2_candidate = marked_text.find(
                                ent2_text, current_search_offset
                            )
                            if pos_e2_candidate == -1:
                                break  # ent2_text not found further

                            # Check if this candidate is part of an already inserted E1 marker block
                            # A simple check: is it immediately preceded by E1_marker_open and followed by E1_marker_close?
                            # This gets complicated quickly. The training script's `replace(text, new, 1)` is what we must emulate.
                            # `replace(text, new, 1)` finds the first occurrence of `text` and replaces it.

                            # For the second replacement: `marked_text.replace(ent2_text, f"[E2] {ent2_text} [/E2]", 1)`
                            # This was the original second part of the chain.

                            # If ent1_text and ent2_text are identical:
                            # stext = "aspirin induces aspirin" (e1=first aspirin, e2=second aspirin)
                            # m1 = stext.replace("aspirin", "[E1] aspirin [/E1]", 1) -> "[E1] aspirin [/E1] induces aspirin"
                            # m2 = m1.replace("aspirin", "[E2] aspirin [/E2]", 1) -> "[E1] aspirin [/E1] induces [E2] aspirin [/E2]"
                            # This is the behavior to replicate.

                            # So, the direct chained replacement is what the model learned.
                            # The `marked_text` already has E1. Now mark E2 on this `marked_text`.
                            break  # Exit loop, use the current `marked_text` for the second replace.

                        # Perform the second replacement on the string that may already contain E1 markers
                        temp_marked_text_for_e2 = marked_text  # string after E1 marking
                        pos_e2_final = temp_marked_text_for_e2.find(
                            ent2_text
                        )  # find ent2_text in it

                        if pos_e2_final != -1:
                            # Check if this is inside the E1 markers we just added
                            # This check helps if e2_text is a substring of e1_text or e1's markers
                            # e.g. e1_text = "aspirin C", e2_text = "aspirin"
                            # marked_text after e1 = "[E1] aspirin C [/E1]"
                            # find("aspirin") would find it inside. We want to avoid replacing part of e1's markers.
                            # The training .replace would correctly replace the "aspirin" inside "[E1] aspirin C [/E1]"
                            # if that was the *first* occurrence of "aspirin" in that string.
                            # This implies the entity texts themselves should not contain "[E1]", "[/E1]" etc.

                            # The most straightforward interpretation of the training script's
                            # `...replace(ent1,...).replace(ent2,...)` is just a sequential operation.
                            marked_text = temp_marked_text_for_e2.replace(
                                ent2_text,
                                f"{e2_marker_open} {ent2_text} {e2_marker_close}",
                                1,
                            )

                        else:
                            # This case should ideally not happen
                            print(
                                f"Warning: Entity text '{ent2_text}' not found in (E1-marked) sentence for E2 marking. Sentence: {stext_original}, E1-marked: {temp_marked_text_for_e2}"
                            )

                    parsed_examples.append(
                        {
                            "sid": s_id,
                            "e1_id": e1_attr_id,
                            "e2_id": e2_attr_id,
                            "text": marked_text,
                        }
                    )
        except FileNotFoundError:
            print(f"Error: File not found {filepath}")
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}")

    return parsed_examples


def main():
    parser = argparse.ArgumentParser(description="Run BERT DDI prediction.")
    parser.add_argument(
        "model_dir",
        help="Directory where the fine-tuned model, tokenizer, and label mappings are saved.",
    )
    parser.add_argument(
        "input_data_path",
        help="Directory with XML files or a single XML file for prediction.",
    )
    parser.add_argument(
        "output_file", help="File to save predictions in sid|e1_id|e2_id|type format."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Batch size for prediction. Should match or be compatible with training if possible, but not strictly necessary for inference VRAM.",
    )
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    try:
        tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
        # Ensure new special tokens are known to the tokenizer if they were added during training
        # The training script adds them and then resizes token embeddings.
        # Loading from the directory should preserve this tokenizer configuration.
        # special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
        # tokenizer.add_special_tokens(special_tokens) # Usually not needed if saved tokenizer_config.json is correct

        model = BertForSequenceClassification.from_pretrained(args.model_dir)
        # model.resize_token_embeddings(len(tokenizer)) # Should not be needed if model was saved after resizing
        model.to(device)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model or tokenizer from {args.model_dir}: {e}")
        return

    # Load id2label mapping
    id2label_path = os.path.join(args.model_dir, "id2label.json")
    try:
        with open(id2label_path, "r") as f:
            id2label_str_keys = json.load(f)
            id2label = {
                int(k): v for k, v in id2label_str_keys.items()
            }  # Convert keys to int
    except FileNotFoundError:
        print(f"Error: id2label.json not found in {args.model_dir}")
        return
    except Exception as e:
        print(f"Error loading id2label.json: {e}")
        return

    # Parse DDI data for prediction
    # This list contains dicts with 'sid', 'e1_id', 'e2_id', 'text'
    prediction_examples = parse_ddi_data_for_prediction(args.input_data_path)

    if not prediction_examples:
        print("No examples found for prediction.")
        return

    # Texts to tokenize and predict
    texts_for_prediction = [example["text"] for example in prediction_examples]

    print(f"Predicting on {len(texts_for_prediction)} examples...")

    with open(args.output_file, "w") as outf:
        # Process in batches
        for i in range(0, len(texts_for_prediction), args.batch_size):
            batch_texts = texts_for_prediction[i : i + args.batch_size]
            batch_examples_info = prediction_examples[
                i : i + args.batch_size
            ]  # Corresponding metadata

            if not batch_texts:  # Should not happen if outer loop condition is correct
                continue

            # Tokenize the batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,  # Pad to longest in batch
                truncation=True,  # Truncate to model max length
                max_length=(
                    tokenizer.model_max_length
                    if hasattr(tokenizer, "model_max_length")
                    else 512
                ),  # Bert max length
            )

            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Perform inference
            with torch.no_grad():  # Important: disable gradient calculations
                outputs = model(**inputs)
                logits = outputs.logits

            predicted_class_indices = torch.argmax(logits, dim=-1).cpu().tolist()

            # Write predictions for the current batch
            for idx, example_info in enumerate(batch_examples_info):
                predicted_label_id = predicted_class_indices[idx]
                predicted_label_str = id2label.get(
                    predicted_label_id, "unknown_label"
                )  # Default if ID somehow not in map

                if predicted_label_str != "null":
                    outf.write(
                        f"{example_info['sid']}|{example_info['e1_id']}|{example_info['e2_id']}|{predicted_label_str}\n"
                    )

    print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
