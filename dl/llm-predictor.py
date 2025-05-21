#!
#!/usr/bin/env python3
"""
llm_predictor_ddi_ollama.py

This script uses a locally running LLM via Ollama to classify 
drug-drug interaction types from XML data using a zero-shot prompting approach.

Dependencies:
    pip install requests tqdm
    Requires dataset.py to be in the same directory or Python path.
    Ollama installed and running with the specified model downloaded.
"""

import argparse
import requests  # For making HTTP requests to Ollama API
import json
import os
from tqdm import tqdm
import sys
import time

# Attempt to import RawDataset from the user's dataset.py
try:
    from dataset import Dataset as RawDataset

    # from dataset import SentenceDict # Optional, for type hinting
except ImportError:
    print(
        "ERROR: Could not import RawDataset from dataset.py. "
        "Ensure dataset.py is in the same directory or accessible in your PYTHONPATH.",
        file=sys.stderr,
    )
    sys.exit(1)


# --- DDI Type Definitions (for the prompt) ---
DDI_DEFINITIONS = {
    "advise": """
Definition: the interaction statement carries a recommendation or
warning—what should or should not be done (contraindications, precautions,
“use with caution,” etc.).
Examples:
– “Nonsteroidal Antiinflammatory Agents: Aspirin is contraindicated
in patients who are hypersensitive to nonsteroidal anti‐inflammatory
agents.”
Entities: (Aspirin ↔ nonsteroidal anti‐inflammatory agents)
type="advise"
– “Enteric Coated Aspirin should not be given concurrently with
antacids…”
Entities: (Aspirin ↔ antacids)
type="advise"
– “Evidence of spontaneous recovery from succinylcholine should be
observed before the administration of MIVACRON.”
Entities: (succinylcholine ↔ MIVACRON)
type="advise"
    """,
    "effect": """
Definition: one drug alters the magnitude of action or
concentration of another.  These statements use verbs like “increase,”
“decrease,” “potentiate,” “block,” or “enhance.”
Examples:
– “Aspirin may decrease the effects of probenecid, sulfinpyrazone,
and phenylbutazone.”
Entities: (Aspirin ↔ probenecid) … (Aspirin ↔ phenylbutazone)
type="effect"
– “FLEXERIL may enhance the effects of alcohol, barbiturates, and
other CNS depressants.”
Entities: (FLEXERIL ↔ alcohol), (FLEXERIL ↔ barbiturates),
(FLEXERIL ↔ CNS depressants)
type="effect"
– “Prior administration of succinylcholine can potentiate the
neuromuscular blocking effects of nondepolarizing agents.”
Entities: (succinylcholine ↔ nondepolarizing agents)
type="effect"
    """,
    "int": """
Definition: a general notice that an interaction exists (or may
exist), but without specifying its direction (increase/decrease), mechanism,
or any particular recommendation.  Often it’s a headline (“X may have
life‐threatening interactions with Y”) or a “possibility of interaction”
statement.
Examples:
– “FLEXERIL may have life-threatening interactions with MAO
inhibitors.”
Entities: (FLEXERIL ↔ MAO inhibitors)
type="int"
– “Animal data have suggested the possibility of interaction
between perindopril and gentamicin.”
Entities: (perindopril ↔ gentamicin)
type="int"
– “Other drugs which may enhance the neuromuscular blocking action
of nondepolarizing agents such as MIVACRON include certain antibiotics…,
magnesium salts, lithium, local anesthetics, procainamide, and quinidine.”
Entities: (nondepolarizing agents ↔ each listed antibiotic, salt,
etc.)
type="int"
    """,
    "mechanism": """
Definition: the text explicitly cites the biochemical or
physiological mechanism behind the interaction (enzyme induction,
competition for receptors or protein‐binding, altered absorption, etc.).
Examples:
– “Phenobarbital: Decreases aspirin effectiveness by enzyme
induction.”
Entities: (Phenobarbital ↔ aspirin)
type="mechanism"
– “Corticosteroids: Concomitant administration with aspirin may
increase the risk of gastrointestinal ulceration and may reduce serum
salicylate levels.”
    """,
    "none": """
1. Definition of “none”
• Sentences are exhaustively paired across all mentions.
• In practice “none” covers co-mentions that share a sentence but for which the text does not assert an effect,
mechanism or advice.
2. Why it’s so tricky
– Entity co-occurrence alone is _not_ enough. Two drugs can appear in the same sentence without interacting.
– Interaction verbs or cautionary language almost always link only one or two specific pairs; _all_ the other pairs in
that sentence belong to None.
– Because a single sentence with N mentions leads to N·(N–1)/2 pairs, the vast majority of pairs are “None.”  A naive
co-occurrence classifier will predict massive false positives.
3. Walkthrough of real examples

A. Aspirin & Uricosuric Agents
“Uricosuric Agents: Aspirin may decrease the effects of probenecid, sulfinpyrazone, and phenylbutazone.”

Entities:
e0 = Uricosuric Agents
e1 = Aspirin
e2 = probenecid
e3 = sulfinpyrazone
e4 = phenylbutazone

All 10 possible pairs, with their gold labels:
• (e0,e1) false → none
• (e0,e2) false → none
• (e0,e3) false → none
• (e0,e4) false → none
• (e1,e2) true type=effect
• (e1,e3) true type=effect
• (e1,e4) true type=effect
• (e2,e3) false → none
• (e2,e4) false → none
• (e3,e4) false → none

Here you see 7 “None” vs. 3 “effect” pairs.

B. FLEXERIL & CNS-Drugs
“FLEXERIL may enhance the effects of alcohol, barbiturates, and other CNS depressants.”

Entities:
e0 = FLEXERIL
e1 = alcohol
e2 = barbiturates
e3 = CNS depressants

Pairs:
• (e0,e1) effect
• (e0,e2) effect
• (e0,e3) effect
• (e1,e2) false → none
• (e1,e3) false → none
• (e2,e3) false → none

Three None pairs vs. three effect pairs.

C. MIVACRON & a dozen co-drugs
“Other drugs which may enhance the neuromuscular blocking action of nondepolarizing agents such as MIVACRON include certain
antibiotics (aminoglycosides, tetracyclines, … colistin, sodium colistimethate), magnesium salts, lithium, local anesthetics,
procainamide, and quinidine.”

Entities:
e0 = nondepolarizing agents (the class)
e1 = aminoglycosides
e2 = tetracyclines
…
e7 = colistin
e8 = sodium colistimethate
e9 = magnesium
e10 = lithium
e11 = local anesthetics
e12 = procainamide
e13 = quinidine

Pairs:
– (e0, e1…e13)  → all true type=int
– Every pair among (e1…e13) → false → None

Concretely, C(13,2)=78 None pairs, and 13 “int” pairs.  Without careful signal-detection you’ll drown in those 78 false
positives.

D. Pyrazolone Derivatives & Aspirin
“Pyrazolone Derivatives (phenylbutazone, oxyphenbutazone, and possibly dipyrone): Concomitant administration with aspirin may
increase the risk…”

Entities:
e0 = Pyrazolone Derivatives
e1 = phenylbutazone
e2 = oxyphenbutazone
e3 = dipyrone
e4 = aspirin

Pairs (10 total):
• (e0,e1) false → none
• (e0,e2) false → none
• (e0,e3) false → none
• (e0,e4) effect
• (e1,e2) false → none
• (e1,e3) false → none
• (e1,e4) effect
• (e2,e3) false → none
• (e2,e4) effect
• (e3,e4) effect

Here 6 none vs. 4 effect.

E. Digoxin & ACEON Tablets
“Digoxin: A controlled study has shown no effect on plasma digoxin concentrations when coadministered with ACEON Tablets, but
an effect of digoxin on the plasma concentration of perindopril/perindoprilat has not been excluded.”

Entities:
e0 = Digoxin (the drug name in headline)
e1 = digoxin (in “plasma digoxin…”)
e2 = ACEON (brand)
e3 = digoxin (again, as substrate)
e4 = perindopril
e5 = perindoprilat

Pairs (15 total):
– Only (e3,e4) and (e3,e5) are true type=mechanism
– All other 13 pairs are false → none

1. Summary & best practice
– **None** = ddi="false" pairs.
– They _dominate_ the data—often 70–90% of all pairs in a document.
– **Tricky**: simple “two drugs in one sentence” features will trigger most of these as false positives.  You _must_
teach your model to spot the *linguistic cue* (verbs like increase/decrease, “by…”, “contraindicated”, “may interact”, etc.)
that locally _links_ exactly the right entities, and ignore all other co-mentions.
– In sequence sentences (e.g. lists of drugs), it is especially easy to over-label every pair—so explicit negative
examples (“None”) of every unintended pair are essential in training.
    """,
}
VALID_LABELS = ["advise", "effect", "int", "mechanism", "none"]
DEFAULT_OLLAMA_MODEL = (
    "gemma:2b-instruct"  # Example, user should have this model in Ollama
)


class LLMPredictorOllama:
    def __init__(
        self,
        ollama_model_name=DEFAULT_OLLAMA_MODEL,
        ollama_api_url="http://localhost:11434/api/generate",
    ):
        self.ollama_model_name = ollama_model_name
        self.ollama_api_url = ollama_api_url
        print(
            f"Initializing LLMPredictorOllama with model: {ollama_model_name} at {ollama_api_url}",
            file=sys.stderr,
        )
        # Test connection or model availability (optional)
        try:
            # Check if the model exists in ollama list, or just proceed and let predict handle errors
            # For simplicity, we'll let `predict` handle API errors.
            pass
        except Exception as e:
            print(
                f"Warning: Could not verify Ollama setup during init: {e}",
                file=sys.stderr,
            )

    def predict(
        self,
        prompt: str,
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a prediction for the given prompt using the Ollama API.
        """
        payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "stream": False,  # Get the full response at once
            "options": {
                "num_predict": max_new_tokens,  # Ollama's way to limit output length
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        # Adjust options for deterministic output if temperature is very low
        if temperature <= 0.001:
            payload["options"]["temperature"] = 0.0  # Explicitly greedy
            if "top_p" in payload["options"]:  # top_p not used with temp 0
                del payload["options"]["top_p"]
            # Ollama might also have a specific way to ensure greedy, temp 0 is usually it.

        try:
            response = requests.post(self.ollama_api_url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            response_data = response.json()

            generated_text = response_data.get("response", "").strip()
            return generated_text

        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}", file=sys.stderr)
            return ""  # Return empty string on error
        except json.JSONDecodeError:
            print(
                f"Error decoding JSON response from Ollama: {response.text}",
                file=sys.stderr,
            )
            return ""


def parse_xml_for_llm(data_input_path):
    # This function can be reused from the previous script (llm_predictor_ddi_causal.py)
    files_to_process = []
    if os.path.isdir(data_input_path):
        for fname in sorted(os.listdir(data_input_path)):
            if fname.lower().endswith(".xml"):
                files_to_process.append(os.path.join(data_input_path, fname))
    elif os.path.isfile(data_input_path) and data_input_path.lower().endswith(".xml"):
        files_to_process.append(data_input_path)
    else:
        print(
            f"Error: Input path '{data_input_path}' is not a valid directory or .xml file.",
            file=sys.stderr,
        )
        yield from ()  # Return an empty generator
        return

    for filepath in files_to_process:
        try:
            doc = minidom.parse(filepath)
            sentences = doc.getElementsByTagName("sentence")
            for s_node in sentences:
                s_id = s_node.getAttribute("id")
                original_sentence_text = s_node.getAttribute("text")
                entities_in_sentence = {
                    e.getAttribute("id"): e.getAttribute("text")
                    for e in s_node.getElementsByTagName("entity")
                }
                pairs = s_node.getElementsByTagName("pair")
                for p_node in pairs:
                    e1_id, e2_id = p_node.getAttribute("e1"), p_node.getAttribute("e2")
                    drug1_text, drug2_text = entities_in_sentence.get(
                        e1_id
                    ), entities_in_sentence.get(e2_id)
                    if drug1_text and drug2_text:
                        yield {
                            "sid": s_id,
                            "e1_id": e1_id,
                            "e2_id": e2_id,
                            "sentence_text": original_sentence_text,  # Not used in this version's prompt directly
                            "drug1_text": drug1_text,  # Not used in this version's prompt directly
                            "drug2_text": drug2_text,  # Not used in this version's prompt directly
                            "s_obj": s_node,  # Pass the sentence object from RawDataset
                        }
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}", file=sys.stderr)


def construct_ddi_prompt_for_ollama(sentence_with_placeholders):
    """
    Constructs the DDI classification prompt for an Ollama Causal LM,
    using a sentence that already contains <DRUG1> and <DRUG2> placeholders.
    """
    # Ollama models (especially instruct-tuned ones) usually respond well to clear instructions.
    # The prompt structure used for Gemma should work well.
    prompt = (
        "You are an expert biomedical text analyst. Your task is to classify the type of drug-drug interaction (DDI) "
        "between Drug 1 (represented as '<DRUG1>') and Drug 2 (represented as '<DRUG2>') in the given sentence. "
        "The interaction should be specific to the context provided by the sentence.\n\n"
        "The possible DDI types are:\n"
        f"- advise: {DDI_DEFINITIONS['advise']}\n"
        f"- effect: {DDI_DEFINITIONS['effect']}\n"
        f"- int: {DDI_DEFINITIONS['int']}\n"
        f"- mechanism: {DDI_DEFINITIONS['mechanism']}\n"
        f"- none: {DDI_DEFINITIONS['none']}\n\n"
        "Context:\n"
        f'Sentence with placeholders: "{sentence_with_placeholders}"\n\n'
        "Based on this sentence (where '<DRUG1>' marks the first drug of interest and '<DRUG2>' marks the second drug of interest) and the definitions, "
        "what is the DDI type between '<DRUG1>' and '<DRUG2>'? "
        "Respond with only one label from the list: advise, effect, int, mechanism, none.\n"
        "The DDI type is: "
    )
    return prompt


def parse_llm_response(response_text):
    text = response_text.strip().lower()
    for label in VALID_LABELS:
        if text.startswith(label):
            if len(text) == len(label) or (
                len(text) > len(label) and not text[len(label)].isalnum()
            ):
                return label
    for label in VALID_LABELS:
        if text == label:
            return label
    for label in VALID_LABELS:
        if label in text:
            print(
                f"Info: LLM output '{response_text}' matched label '{label}' via substring.",
                file=sys.stderr,
            )
            return label
        if label == "advice":
            return "advise"
    if "no interaction" in text or "no ddi" in text:
        return "none"

    print(
        f"Warning: LLM output '{response_text}' not parsable to a valid label. Defaulting to 'none'.",
        file=sys.stderr,
    )
    return "none"


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based DDI classifier using Ollama and RawDataset."
    )
    parser.add_argument(
        "--ollama_model_name",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model tag (e.g., 'gemma:2b-instruct', 'llama3:8b-instruct') (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--ollama_api_url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama API endpoint URL (default: http://localhost:11434/api/generate)",
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        required=True,
        help="Path to the DDI data (parsed .pck file or XML directory, compatible with RawDataset).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="File to save predictions in sid|e1_id|e2_id|type format.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum number of new tokens for the DDI type label (Ollama's num_predict) (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to process (for testing).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for Ollama generation (0.0 for greedy, >0 for sampling).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling for Ollama generation (used if temperature > 0).",
    )
    parser.add_argument(
        "--generation_delay",
        type=float,
        default=0.1,  # Small delay by default
        help="Delay in seconds between Ollama API calls (default: 0.1).",
    )

    args = parser.parse_args()

    predictor = LLMPredictorOllama(
        ollama_model_name=args.ollama_model_name, ollama_api_url=args.ollama_api_url
    )

    print(
        f"Loading DDI data from: {args.input_data_path} using RawDataset...",
        file=sys.stderr,
    )
    try:
        # RawDataset loads all data into memory with list(raw_data_loader.sentences())
        # This is fine for moderately sized datasets.
        raw_data_loader = RawDataset(args.input_data_path)
        all_examples = list(raw_data_loader.sentences())  # SentenceDict items
    except Exception as e:
        print(
            f"Error loading data with RawDataset from '{args.input_data_path}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.limit is not None:
        all_examples = all_examples[: args.limit]

    if not all_examples:
        print("No examples found from RawDataset.", file=sys.stderr)
        return

    print(
        f"Starting DDI classification for {len(all_examples)} examples using Ollama model {args.ollama_model_name}...",
        file=sys.stderr,
    )

    with open(args.output_file, "w") as outf:
        for example_sentence_dict in tqdm(
            all_examples, desc=f"Classifying with {args.ollama_model_name}"
        ):
            # example_sentence_dict is a SentenceDict from RawDataset
            # example_sentence_dict['sent'] is a list of TokenDicts, e.g., [{'form': 'word1'}, {'form': '<DRUG1>'}, ...]

            sentence_with_placeholders = " ".join(
                [token["form"] for token in example_sentence_dict["sent"]]
            )

            prompt_text = construct_ddi_prompt_for_ollama(sentence_with_placeholders)

            llm_response_text = predictor.predict(
                prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            predicted_label = parse_llm_response(llm_response_text)

            if predicted_label != "none":
                outf.write(
                    f"{example_sentence_dict['sid']}|{example_sentence_dict['e1']}|{example_sentence_dict['e2']}|{predicted_label}\n"
                )

            if args.generation_delay > 0:
                time.sleep(args.generation_delay)

    print(
        f"LLM-based DDI predictions (Ollama) saved to {args.output_file}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
