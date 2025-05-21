#!/usr/bin/env python3
"""
llm_predictor_ddi_causal.py

This script loads a HuggingFace Causal LLM (e.g., Gemma, Mistral)
and uses it to classify drug-drug interaction types from XML data
using a zero-shot prompting approach.

Dependencies:
    pip install transformers torch sentencepiece bitsandbytes accelerate tqdm
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
from xml.dom import minidom
from tqdm import tqdm
import sys  # For stderr

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
    def __init__(
        self, model_name=DEFAULT_CAUSAL_MODEL, device_option=None, quantization=None
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set (common for Causal LMs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(
                f"Tokenizer pad_token_id set to eos_token_id: {self.tokenizer.eos_token_id}",
                file=sys.stderr,
            )

        # Configure quantization
        bnb_config = None
        torch_dtype = torch.float16  # Default for faster loading/inference
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
            device_map="auto",  # Automatically handle device placement
            trust_remote_code=True,  # If model requires custom code
        )

        # Determine pipeline device argument
        if device_option:
            try:
                resolved_device = torch.device(device_option)
                if resolved_device.type == "cuda":
                    if not torch.cuda.is_available():
                        print(
                            f"Warning: Device '{device_option}' requested but CUDA not available. Using CPU.",
                            file=sys.stderr,
                        )
                        self.pipeline_device_arg = -1
                    else:
                        self.pipeline_device_arg = (
                            resolved_device.index
                            if resolved_device.index is not None
                            else 0
                        )
                        print(
                            f"Pipeline explicitly set to GPU device: cuda:{self.pipeline_device_arg}",
                            file=sys.stderr,
                        )
                else:  # CPU
                    self.pipeline_device_arg = -1
                    print(
                        f"Pipeline explicitly set to CPU device: {resolved_device.type}",
                        file=sys.stderr,
                    )
            except RuntimeError:  # If invalid device string
                print(
                    f"Warning: Invalid device string '{device_option}'. Auto-detecting.",
                    file=sys.stderr,
                )
                self.pipeline_device_arg = 0 if torch.cuda.is_available() else -1
        else:  # Auto-detect
            self.pipeline_device_arg = 0 if torch.cuda.is_available() else -1

        if self.pipeline_device_arg >= 0:
            print(
                f"Pipeline will attempt to use GPU: {self.pipeline_device_arg}",
                file=sys.stderr,
            )
        else:
            print("Pipeline will use CPU.", file=sys.stderr)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.pipeline_device_arg,
        )

    def predict(self, prompt: str, max_new_tokens: int = 10, **generate_kwargs) -> str:
        """
        Generate a prediction for the given prompt using a text-generation model.
        """
        # Default generate_kwargs for more deterministic output
        final_generate_kwargs = {
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": False,  # Default to greedy for classification
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        # Override defaults with any user-provided kwargs
        final_generate_kwargs.update(generate_kwargs)

        # If temperature is set > 0 by user, enable sampling
        if (
            final_generate_kwargs.get("temperature", 0) > 0.001
        ):  # Check if temperature is meaningfully > 0
            final_generate_kwargs["do_sample"] = True
        else:  # Ensure deterministic if temp is ~0
            final_generate_kwargs["do_sample"] = False
            final_generate_kwargs.pop("top_p", None)  # top_p not used with greedy
            final_generate_kwargs.pop(
                "temperature", None
            )  # temperature not used with greedy

        # The text-generation pipeline returns a list of dictionaries.
        # Each dictionary has 'generated_text' which includes the prompt.
        outputs = self.pipeline(
            prompt, max_new_tokens=max_new_tokens, **final_generate_kwargs
        )

        full_generated_text = outputs[0]["generated_text"]
        # Extract only the generated part (after the prompt)
        # This assumes the prompt is faithfully reproduced at the start of generated_text
        if full_generated_text.startswith(prompt):
            return full_generated_text[len(prompt) :].strip()
        else:
            # Fallback if prompt is not exactly at the start (can happen with some models/pipelines)
            # This is a heuristic. A more robust way might involve passing input_ids to generate
            # and then decoding only the output_ids that extend beyond input_ids.
            # However, the pipeline abstracts this.
            print(
                f"Warning: Prompt not found at the beginning of generated text. Output: '{full_generated_text[:200]}...'",
                file=sys.stderr,
            )
            # Attempt to return the last part, hoping it's the answer
            lines = full_generated_text.splitlines()
            return lines[-1].strip() if lines else ""


def parse_xml_for_llm(data_input_path):
    # This function can be reused from the previous script (llm_predictor_ddi.py)
    # For brevity, I'll assume it's available or copy-paste it here if needed.
    # It yields dicts with: sid, e1_id, e2_id, sentence_text, drug1_text, drug2_text
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
                            "sentence_text": original_sentence_text,
                            "drug1_text": drug1_text,
                            "drug2_text": drug2_text,
                        }
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}", file=sys.stderr)


def construct_ddi_prompt_for_causal_lm(sentence_text, drug1_text, drug2_text):
    """Constructs the DDI classification prompt for a Causal LM."""
    # For instruction-tuned Causal LMs, a clear instruction format is good.
    # The prompt should end in a way that the model naturally completes with the label.
    prompt = (
        "You are an expert biomedical text analyst. Your task is to classify the type of drug-drug interaction (DDI) "
        "between two specified drugs in the given sentence. The interaction should be specific to the context provided.\n\n"
        "The possible DDI types are:\n"
        f"- advise: {DDI_DEFINITIONS['advise']}\n"
        f"- effect: {DDI_DEFINITIONS['effect']}\n"
        f"- int: {DDI_DEFINITIONS['int']}\n"
        f"- mechanism: {DDI_DEFINITIONS['mechanism']}\n"
        f"- none: {DDI_DEFINITIONS['none']}\n\n"
        "Context:\n"
        f'Sentence: "{sentence_text}"\n'
        f'Drug 1: "{drug1_text}"\n'
        f'Drug 2: "{drug2_text}"\n\n'
        "Based on the sentence and definitions, what is the DDI type? "
        "Respond with only one label from the list: advise, effect, int, mechanism, none.\n"
        "The DDI type is: "  # This ending encourages the model to complete with the label.
    )
    return prompt


def parse_llm_response(response_text):
    # This function can be reused from the previous script (llm_predictor_ddi.py)
    text = response_text.strip().lower()
    for label in VALID_LABELS:  # Exact match first
        if text == label:
            return label
    for label in VALID_LABELS:  # Substring match
        if label in text:
            return label
    if "no interaction" in text or "no ddi" in text:
        return "none"
    print(
        f"Warning: LLM output '{response_text}' not parsable. Defaulting to 'none'.",
        file=sys.stderr,
    )
    return "none"


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based DDI classifier using Causal LM (text-generation)."
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
        help="Directory with XML files or a single XML file for DDI classification.",
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
        default=10,  # Expecting very short label outputs
        help="Maximum number of new tokens to generate for the DDI type label (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to process (for testing).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (e.g., 'cpu', 'cuda', 'cuda:0'). Default: auto-detect by pipeline.",
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
        default=0.0,  # Default to 0 for more deterministic output
        help="Temperature for LLM generation (0.0 for greedy, >0 for sampling).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling for LLM generation (used if temperature > 0).",
    )
    # num_beams could be added, but greedy/sampling is often sufficient for this.

    args = parser.parse_args()

    predictor = LLMPredictorCausal(
        model_name=args.model_name,
        device_option=args.device,
        quantization=args.quantization,
    )

    parsed_examples = list(parse_xml_for_llm(args.input_data_path))
    if args.limit is not None:
        parsed_examples = parsed_examples[: args.limit]

    if not parsed_examples:
        print("No examples found for prediction.", file=sys.stderr)
        return

    print(
        f"Starting DDI classification for {len(parsed_examples)} examples using {args.model_name}...",
        file=sys.stderr,
    )

    generate_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        # "num_beams": 1 # Default to greedy/sampling
    }

    with open(args.output_file, "w") as outf:
        for example in tqdm(
            parsed_examples,
            desc=f"Classifying with {os.path.basename(args.model_name)}",
        ):
            prompt_text = construct_ddi_prompt_for_causal_lm(
                example["sentence_text"], example["drug1_text"], example["drug2_text"]
            )

            llm_response_text = predictor.predict(
                prompt_text, max_new_tokens=args.max_new_tokens, **generate_kwargs
            )

            predicted_label = parse_llm_response(llm_response_text)

            if predicted_label != "none":
                outf.write(
                    f"{example['sid']}|{example['e1_id']}|{example['e2_id']}|{predicted_label}\n"
                )

    print(f"LLM-based DDI predictions saved to {args.output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
