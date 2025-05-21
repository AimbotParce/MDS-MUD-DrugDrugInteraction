#!/usr/bin/env
#!/usr/bin/env python3
"""
llm_predictor_ddi_causal.py

This script loads a HuggingFace Causal LLM (e.g., Gemma)
and uses it to classify drug-drug interaction types. It uses RawDataset
to load DDI data, where sentences already have <DRUG1> and <DRUG2> placeholders.

Dependencies:
    pip install transformers torch sentencepiece bitsandbytes accelerate tqdm
    Requires dataset.py to be in the same directory or Python path.
"""

import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import os

# from xml.dom import minidom # Not needed if using RawDataset
from tqdm import tqdm
import sys
import time

# Attempt to import RawDataset from the user's dataset.py
try:
    from dataset import Dataset as RawDataset
    from dataset import SentenceDict  # Optional, for type hinting if needed
except ImportError:
    print(
        "ERROR: Could not import RawDataset from dataset.py. "
        "Ensure dataset.py is in the same directory or accessible in your PYTHONPATH.",
        file=sys.stderr,
    )
    sys.exit(1)


# --- DDI Type Definitions (for the prompt) ---
DDI_DEFINITIONS = {
    "advise": "An interaction where caution, monitoring, or a change in therapy is advised when the drugs are co-administered (e.g., 'The combination requires regular monitoring').",
    "effect": "An interaction where a pharmacodynamic or pharmacokinetic effect is described, often altering the drug's efficacy or leading to side effects (e.g., 'Drug A significantly reduces plasma concentrations of Drug B', 'Concurrent use increases risk of bleeding').",
    "int": "An intended or synergistic interaction where the combination is therapeutically beneficial (e.g., 'Drug A is co-administered with Drug B to enhance its bioavailability').",
    "mechanism": "An interaction where the underlying biological or chemical mechanism is described (e.g., 'Drug A inhibits the metabolism of Drug B via CYP3A4 inhibition', 'Drug A induces enzymes that metabolize Drug B').",
    "none": "No functional interaction is described between the two specified drugs in the given context, or the interaction does not fit any of the other categories.",
}
VALID_LABELS = ["advise", "effect", "int", "mechanism", "none"]
DEFAULT_CAUSAL_MODEL = "google/gemma-2b-it"


class LLMPredictorCausal:
    def __init__(self, model_name=DEFAULT_CAUSAL_MODEL, quantization=None):
        self.model_name = model_name
        print(
            f"Initializing LLMPredictorCausal with model: {model_name}", file=sys.stderr
        )
        print(f"Quantization: {quantization}", file=sys.stderr)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(
                f"Tokenizer pad_token_id set to eos_token_id: {self.tokenizer.eos_token_id}",
                file=sys.stderr,
            )

        bnb_config = None
        torch_dtype = torch.float16
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            torch_dtype = torch.bfloat16
            print("Loading model with 4-bit quantization.", file=sys.stderr)
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            print("Loading model with 8-bit quantization.", file=sys.stderr)
        else:
            print(f"Loading model in {torch_dtype} precision.", file=sys.stderr)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        print(
            f"Model loaded. Device of model params: {next(self.model.parameters()).device}",
            file=sys.stderr,
        )

        self.pipeline = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )
        print("Text-generation pipeline initialized.", file=sys.stderr)

    def predict(self, prompt: str, max_new_tokens: int = 10, **generate_kwargs) -> str:
        final_generate_kwargs = {
            "temperature": 0.1,  # Low temperature for more deterministic output
            "top_p": 0.9,
            "do_sample": False,  # Default to greedy for classification
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        final_generate_kwargs.update(generate_kwargs)

        if (
            final_generate_kwargs.get("temperature", 0) > 0.001
        ):  # Check if temperature is meaningfully > 0
            final_generate_kwargs["do_sample"] = True
        else:
            final_generate_kwargs["do_sample"] = False
            final_generate_kwargs.pop("top_p", None)
            final_generate_kwargs.pop("temperature", None)

        outputs = self.pipeline(
            prompt, max_new_tokens=max_new_tokens, **final_generate_kwargs
        )

        full_generated_text = outputs[0]["generated_text"]
        if full_generated_text.startswith(prompt):  # Common case
            return full_generated_text[len(prompt) :].strip()
        else:
            # Fallback if prompt is not exactly at the start
            # This can happen if the model adds a BOS token or reformats slightly
            # We try to return the part that looks like the answer
            # Often, the answer is the last significant part of the string.
            lines = full_generated_text.splitlines()
            potential_answer = lines[-1].strip() if lines else ""
            # If the potential answer is very long, it might be the model repeating parts of the prompt.
            # This is a heuristic.
            if (
                len(potential_answer) > max_new_tokens * 2
                and len(potential_answer) > len(prompt) / 2
            ):
                print(
                    f"Warning: Generated text did not start with prompt and fallback is long. Prompt: '{prompt[:100]}...' Output: '{full_generated_text[:200]}...'",
                    file=sys.stderr,
                )
                # Try to find the instruction part of the prompt in the output
                instruction_phrase = "The DDI type is: "
                idx = full_generated_text.rfind(instruction_phrase)
                if idx != -1:
                    return full_generated_text[idx + len(instruction_phrase) :].strip()
                return potential_answer  # Or just return the raw output for manual inspection
            return potential_answer


def construct_ddi_prompt_for_causal_lm_with_placeholders(sentence_with_placeholders):
    """
    Constructs the DDI classification prompt for a Causal LM,
    using a sentence that already contains <DRUG1> and <DRUG2> placeholders.
    """
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
    if "no interaction" in text or "no ddi" in text:
        return "none"

    print(
        f"Warning: LLM output '{response_text}' not parsable to a valid label. Defaulting to 'none'.",
        file=sys.stderr,
    )
    return "none"


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based DDI classifier using Causal LM and RawDataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_CAUSAL_MODEL,
        help=f"HuggingFace Causal LM name (default: {DEFAULT_CAUSAL_MODEL})",
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
        help="Maximum number of new tokens for the DDI type label (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to process (for testing).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="Enable quantization: '4bit' or '8bit'. Requires bitsandbytes and accelerate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation (0.0 for greedy, >0 for sampling).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling for LLM generation (used if temperature > 0).",
    )
    parser.add_argument(
        "--generation_delay",
        type=float,
        default=0.0,
        help="Delay in seconds between LLM inferences (default: 0.0).",
    )

    args = parser.parse_args()

    predictor = LLMPredictorCausal(
        model_name=args.model_name, quantization=args.quantization
    )

    # Load data using RawDataset
    print(
        f"Loading DDI data from: {args.input_data_path} using RawDataset...",
        file=sys.stderr,
    )
    try:
        raw_data_loader = RawDataset(args.input_data_path)
        # Convert generator to list to apply limit and use tqdm
        all_examples = list(raw_data_loader.sentences())
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
        f"Starting DDI classification for {len(all_examples)} examples using {args.model_name}...",
        file=sys.stderr,
    )

    generate_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    with open(args.output_file, "w") as outf:
        for example_sentence_dict in tqdm(
            all_examples, desc=f"Classifying with {os.path.basename(args.model_name)}"
        ):
            # example_sentence_dict is a SentenceDict from RawDataset
            # s['sent'] is a list of TokenDicts, e.g., [{'form': 'word1'}, {'form': '<DRUG1>'}, ...]

            sentence_with_placeholders = " ".join(
                [token["form"] for token in example_sentence_dict["sent"]]
            )

            prompt_text = construct_ddi_prompt_for_causal_lm_with_placeholders(
                sentence_with_placeholders
            )

            llm_response_text = predictor.predict(
                prompt_text, max_new_tokens=args.max_new_tokens, **generate_kwargs
            )

            predicted_label = parse_llm_response(llm_response_text)

            if predicted_label != "none":
                outf.write(
                    f"{example_sentence_dict['sid']}|{example_sentence_dict['e1']}|{example_sentence_dict['e2']}|{predicted_label}\n"
                )

            if args.generation_delay > 0:
                time.sleep(args.generation_delay)

    print(f"LLM-based DDI predictions saved to {args.output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
