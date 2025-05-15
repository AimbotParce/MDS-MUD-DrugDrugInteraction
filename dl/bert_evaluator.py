#!/usr/bin/env python3

import argparse
import os
from xml.dom import minidom
from collections import defaultdict


def load_gold_relations(gold_input_path):
    """
    Loads gold standard DDI relations from XML files.

    Args:
        gold_input_path (str): Path to a directory of XML files or a single XML file.

    Returns:
        tuple: A set of gold relations (sid, e1_id, e2_id, type) and
               a sorted list of unique DDI types found.
    """
    gold_relations = set()
    label_types = set()
    files_to_process = []

    if os.path.isdir(gold_input_path):
        for fname in sorted(os.listdir(gold_input_path)):
            if fname.lower().endswith(".xml"):
                files_to_process.append(os.path.join(gold_input_path, fname))
    elif os.path.isfile(gold_input_path) and gold_input_path.lower().endswith(".xml"):
        files_to_process.append(gold_input_path)
    else:
        raise ValueError(
            f"Gold data input '{gold_input_path}' must be a directory of XML files or a single XML file."
        )

    for filepath in files_to_process:
        try:
            doc = minidom.parse(filepath)
            sentences = doc.getElementsByTagName("sentence")
            for s_node in sentences:
                s_id = s_node.getAttribute("id")
                pairs = s_node.getElementsByTagName("pair")
                for p_node in pairs:
                    is_ddi = p_node.getAttribute("ddi")
                    if is_ddi.lower() == "true":
                        e1_id = p_node.getAttribute("e1")
                        e2_id = p_node.getAttribute("e2")
                        ddi_type = p_node.getAttribute("type")

                        if not ddi_type:
                            print(
                                f"Warning: Missing type for DDI pair ({e1_id}, {e2_id}) in sentence {s_id} of file {filepath}. Skipping."
                            )
                            continue
                        if not e1_id or not e2_id:
                            print(
                                f"Warning: Missing e1 or e2 ID for DDI pair in sentence {s_id} of file {filepath}. Type was '{ddi_type}'. Skipping."
                            )
                            continue

                        gold_relations.add((s_id, e1_id, e2_id, ddi_type))
                        label_types.add(ddi_type)
        except Exception as e:
            print(f"Error parsing gold file {filepath}: {e}")

    return gold_relations, sorted(list(label_types))


def load_predicted_relations(prediction_file):
    """
    Loads predicted DDI relations from a file.
    Expected format: sid|e1_id|e2_id|type (one per line)

    Args:
        prediction_file (str): Path to the prediction file.

    Returns:
        set: A set of predicted relations (sid, e1_id, e2_id, type).
    """
    predicted_relations = set()
    try:
        with open(prediction_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) == 4:
                    s_id, e1_id, e2_id, ddi_type = parts
                    # Predictions of "null" type are typically not evaluated as positive instances.
                    # The prediction script should ideally filter these out already.
                    # If they are present, they will count as FPs if not in gold,
                    # or if "null" was a gold type (which it shouldn't be for true DDIs).
                    if ddi_type.lower() == "null":
                        continue  # Do not count "null" predictions towards evaluated metrics
                    predicted_relations.add((s_id, e1_id, e2_id, ddi_type))
                else:
                    print(
                        f"Warning: Malformed line {i+1} in prediction file: '{line}'. Expected 4 pipe-separated parts."
                    )
    except FileNotFoundError:
        print(f"Error: Prediction file '{prediction_file}' not found.")
        return set()  # Return empty set if file not found
    except Exception as e:
        print(f"Error reading prediction file {prediction_file}: {e}")
        return set()

    return predicted_relations


def calculate_metrics(tp, fp, fn):
    """Calculates precision, recall, and F1-score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def evaluate_ddi(gold_relations, predicted_relations, all_ddi_types):
    """
    Compares gold and predicted DDI relations and prints evaluation metrics.
    """
    print("\n--- Overall Evaluation (Micro-average) ---")

    true_positives_overall = len(gold_relations.intersection(predicted_relations))
    false_positives_overall = len(predicted_relations - gold_relations)
    false_negatives_overall = len(gold_relations - predicted_relations)

    precision_overall, recall_overall, f1_overall = calculate_metrics(
        true_positives_overall, false_positives_overall, false_negatives_overall
    )

    print(f"  True Positives (TP): {true_positives_overall}")
    print(f"  False Positives (FP): {false_positives_overall}")
    print(f"  False Negatives (FN): {false_negatives_overall}")
    print(f"  Precision: {precision_overall:.4f}")
    print(f"  Recall:    {recall_overall:.4f}")
    print(f"  F1-score:  {f1_overall:.4f}")

    print("\n--- Per-type Evaluation ---")
    if not all_ddi_types:
        print("  No DDI types found in gold standard for per-type evaluation.")
        return

    for ddi_type in all_ddi_types:
        print(f"------------------------------\n  Type: {ddi_type}")

        gold_for_type = {rel for rel in gold_relations if rel[3] == ddi_type}
        predicted_for_type = {rel for rel in predicted_relations if rel[3] == ddi_type}

        tp_type = len(gold_for_type.intersection(predicted_for_type))
        fp_type = len(predicted_for_type - gold_for_type)
        fn_type = len(gold_for_type - predicted_for_type)

        p_type, r_type, f1_type = calculate_metrics(tp_type, fp_type, fn_type)

        print(f"    TP: {tp_type}, FP: {fp_type}, FN: {fn_type}")
        print(f"    Precision: {p_type:.4f}")
        print(f"    Recall:    {r_type:.4f}")
        print(f"    F1-score:  {f1_type:.4f}")
    print("------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDI extraction performance.")
    parser.add_argument(
        "gold_input",
        help="Path to the gold standard DDI data (directory of XMLs or a single XML file).",
    )
    parser.add_argument(
        "prediction_file",
        help="Path to the system's prediction file (format: sid|e1_id|e2_id|type).",
    )

    args = parser.parse_args()

    print(f"Loading gold relations from: {args.gold_input}")
    gold_relations, all_ddi_types = load_gold_relations(args.gold_input)
    if (
        not gold_relations and not all_ddi_types
    ):  # Check if loading failed significantly
        if not os.path.exists(args.gold_input):
            print(f"Error: Gold input path '{args.gold_input}' does not exist.")
        else:
            print(
                "No gold relations loaded. Please check the input path and XML format."
            )
        return

    print(
        f"Found {len(gold_relations)} gold DDI relations of types: {all_ddi_types if all_ddi_types else 'None'}."
    )

    print(f"\nLoading predicted relations from: {args.prediction_file}")
    predicted_relations = load_predicted_relations(args.prediction_file)
    if not predicted_relations and not os.path.exists(args.prediction_file):
        print(
            f"Prediction file '{args.prediction_file}' was not found or is empty. Cannot evaluate."
        )
        # No return here, allows evaluation against empty predictions if file exists but is empty

    print(
        f"Found {len(predicted_relations)} predicted DDI relations (excluding 'null')."
    )

    evaluate_ddi(gold_relations, predicted_relations, all_ddi_types)


if __name__ == "__main__":
    main()
