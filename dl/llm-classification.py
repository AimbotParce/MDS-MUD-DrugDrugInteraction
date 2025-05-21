import argparse
import os
import json
from xml.dom import minidom
from tqdm import tqdm
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- DDI Type Definitions (for the prompt) ---
DDI_DEFINITIONS = {
    "advise": "An interaction where caution, monitoring, or a change in therapy is advised when the drugs are co-administered (e.g., 'The combination requires regular monitoring').",
    "effect": "An interaction where a pharmacodynamic or pharmacokinetic effect is described, often altering the drug's efficacy or leading to side effects (e.g., 'Drug A significantly reduces plasma concentrations of Drug B', 'Concurrent use increases risk of bleeding').",
    "int": "An intended or synergistic interaction where the combination is therapeutically beneficial (e.g., 'Drug A is co-administered with Drug B to enhance its bioavailability').",
    "mechanism": "An interaction where the underlying biological or chemical mechanism is described (e.g., 'Drug A inhibits the metabolism of Drug B via CYP3A4 inhibition', 'Drug A induces enzymes that metabolize Drug B').",
    "none": "No functional interaction is described between the two specified drugs in the given context, or the interaction does not fit any of the other categories.",
}

VALID_LABELS = ["advise", "effect", "int", "mechanism", "none"]


def parse_xml_for_llm(data_input_path):
    """
    Parses DDI XML files to extract sentences and entity pairs for LLM prompting.
    Yields dictionaries, each containing:
    'sid': sentence ID
    'e1_id': ID of the first entity in the pair
    'e2_id': ID of the second entity in the pair
    'sentence_text': original sentence text
    'drug1_text': text of the first drug
    'drug2_text': text of the second drug
    """
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
        return

    for filepath in files_to_process:
        try:
            doc = minidom.parse(filepath)
            sentences = doc.getElementsByTagName("sentence")
            for s_node in sentences:
                s_id = s_node.getAttribute("id")
                original_sentence_text = s_node.getAttribute("text")

                entities_in_sentence = {}
                for e_node in s_node.getElementsByTagName("entity"):
                    entities_in_sentence[e_node.getAttribute("id")] = (
                        e_node.getAttribute("text")
                    )

                pairs = s_node.getElementsByTagName("pair")
                for p_node in pairs:
                    e1_id = p_node.getAttribute("e1")
                    e2_id = p_node.getAttribute("e2")

                    drug1_text = entities_in_sentence.get(e1_id)
                    drug2_text = entities_in_sentence.get(e2_id)

                    if not drug1_text or not drug2_text:
                        continue

                    yield {
                        "sid": s_id,
                        "e1_id": e1_id,
                        "e2_id": e2_id,
                        "sentence_text": original_sentence_text,
                        "drug1_text": drug1_text,
                        "drug2_text": drug2_text,
                    }
        except FileNotFoundError:
            print(f"Error: File not found {filepath}", file=sys.stderr)
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}", file=sys.stderr)


def construct_prompt(sentence_text, drug1_text, drug2_text):
    """Constructs the prompt for the LLM."""
    # Gemma instruction-tuned models expect a specific format.
    # Typically: <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
    # The tokenizer should handle adding <bos> and <eos> if needed based on its config.
    # We will provide the user part of the prompt.

    prompt_content = (
        "You are an expert biomedical text analyst. Your task is to classify the type of drug-drug interaction (DDI) "
        "between two specified drugs in a given sentence. The interaction should be specific to the context provided.\n\n"
        "The possible DDI types are:\n"
        f"- advise: {DDI_DEFINITIONS['advise']}\n"
        f"- effect: {DDI_DEFINITIONS['effect']}\n"
        f"- int: {DDI_DEFINITIONS['int']}\n"
        f"- mechanism: {DDI_DEFINITIONS['mechanism']}\n"
        f"- none: {DDI_DEFINITIONS['none']}\n\n"
        "Consider the following:\n"
        f'Sentence: "{sentence_text}"\n'
        f'Drug 1: "{drug1_text}"\n'
        f'Drug 2: "{drug2_text}"\n\n'
        "What is the DDI type between Drug 1 and Drug 2 in this sentence? "
        "Respond with only one of the following labels: advise, effect, int, mechanism, none.\n"
        "Your response must be only the label."
    )
    # For Gemma instruct models, the prompt is typically formatted like this by the tokenizer's chat template
    # or manually if not using chat template for generation.
    # Let's try a direct prompt and rely on the -it model's ability to follow instructions.
    # If using chat template: messages = [{"role": "user", "content": prompt_content}]
    # tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # For simplicity here, we'll pass the raw prompt_content.
    return prompt_content


def get_hf_llm_prediction(
    model, tokenizer, prompt_text, device, max_new_tokens=10, temperature=0.1, top_p=0.9
):
    """Sends prompt to a local Hugging Face LLM and gets prediction."""
    try:
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length - max_new_tokens,
        ).to(
            device
        )  # Ensure space for new tokens

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,  # Enable sampling for temperature/top_p
                temperature=(
                    temperature if temperature > 0 else None
                ),  # Temp must be > 0 for sampling
                top_p=top_p if temperature > 0 else None,  # top_p only if temp > 0
                pad_token_id=tokenizer.eos_token_id,  # Important for open-ended generation
            )

        # Decode only the newly generated tokens
        response_text = (
            tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            .strip()
            .lower()
        )

        # Simple parsing: check if any valid label is in the response
        for label in VALID_LABELS:
            if (
                label in response_text
            ):  # Check if the label is a substring of the response
                # More robust: check if response_text.startswith(label) or exact match
                # For now, simple substring match
                if response_text.startswith(label) or response_text == label:
                    return label

        # Fallback for less precise matches if the above fails
        for label in VALID_LABELS:
            if label in response_text:
                print(
                    f"Info: LLM output '{response_text}' matched label '{label}' via substring. Consider refining parsing.",
                    file=sys.stderr,
                )
                return label

        if "no interaction" in response_text or "no ddi" in response_text:
            return "none"

        print(
            f"Warning: LLM output '{response_text}' not a recognized label. Defaulting to 'none'. Prompt was: {prompt_text[:100]}...",
            file=sys.stderr,
        )
        return "none"

    except Exception as e:
        print(f"LLM prediction failed: {e}", file=sys.stderr)
        return "none"


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-based DDI classification (Zero-Shot) using local Hugging Face model."
    )
    parser.add_argument(
        "input_data_path",
        help="Directory with XML files or a single XML file for prediction.",
    )
    parser.add_argument(
        "output_file", help="File to save predictions in sid|e1_id|e2_id|type format."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/gemma-3-27b-it",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to process (for testing).",
    )
    parser.add_argument(
        "--api_request_delay",
        type=float,
        default=0.1,
        help="Delay in seconds between inferences (if needed).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=15,
        help="Max new tokens for LLM generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM generation (0.0 for deterministic if do_sample=False).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling for LLM generation.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="Enable quantization: '4bit' or '8bit'. Requires bitsandbytes.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing prompts. Note: current generation loop is 1 by 1.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)
    if args.quantization and device.type == "cpu":
        print(
            "Warning: Quantization is typically for GPU usage. Performance on CPU with quantization might be suboptimal.",
            file=sys.stderr,
        )

    # Configure quantization
    bnb_config = None
    torch_dtype = (
        torch.float16
    )  # Default to float16 for faster loading/inference if not quantizing to int8/4
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16
        )
        torch_dtype = torch.bfloat16  # Recommended compute dtype for 4bit
        print("Loading model with 4-bit quantization.", file=sys.stderr)
    elif args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        # For 8-bit, model will be loaded in 8-bit, compute can be float16 or float32
        print("Loading model with 8-bit quantization.", file=sys.stderr)
    else:
        print(
            f"Loading model in default precision ({torch_dtype}). This may require significant VRAM for large models.",
            file=sys.stderr,
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # Set pad token if not set, common for Gemma and other causal LMs for generation
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(
                f"Tokenizer pad_token_id set to eos_token_id: {tokenizer.eos_token_id}",
                file=sys.stderr,
            )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,  # Important for non-quantized or for compute_dtype with quantization
            device_map="auto",  # Automatically distribute model layers across GPUs if available, or load to CPU if no CUDA
            trust_remote_code=True,  # If model requires custom code
        )
        # No explicit .to(device) needed if device_map="auto" is used and handles it.
        # If not using device_map or it loads to CPU by default and you have a GPU:
        if device.type == "cuda" and (
            bnb_config is None or model.device.type == "cpu"
        ):  # Check if model is not already on GPU
            print(
                f"Manually moving model to {device} (this might be slow for large models without device_map).",
                file=sys.stderr,
            )
            # model.to(device) # This can be very slow and memory intensive for large models if not done by device_map
            # For very large models, device_map is preferred. If it still ends up on CPU,
            # this explicit .to(device) might be needed but could fail for >1 GPU setups or if model is too big.
            # With device_map="auto", Hugging Face handles placement.

        model.eval()  # Set to evaluation mode
    except Exception as e:
        print(
            f"Error loading model or tokenizer '{args.model_name_or_path}': {e}",
            file=sys.stderr,
        )
        print(
            "Make sure you have 'bitsandbytes' and 'accelerate' installed if using quantization: pip install bitsandbytes accelerate",
            file=sys.stderr,
        )
        return

    parsed_examples = list(parse_xml_for_llm(args.input_data_path))

    if args.limit is not None:
        parsed_examples = parsed_examples[: args.limit]

    if not parsed_examples:
        print("No examples found for prediction.", file=sys.stderr)
        return

    print(
        f"Starting DDI classification for {len(parsed_examples)} examples using {args.model_name_or_path}..."
    )

    # Note: Current loop processes one by one. Batching model.generate() for CausalLMs
    # with varying input lengths requires careful padding and attention mask handling,
    # often more complex than for Encoder models.
    # For simplicity, we process individually here.

    with open(args.output_file, "w") as outf:
        for example in tqdm(
            parsed_examples,
            desc=f"Classifying with {os.path.basename(args.model_name_or_path)}",
        ):
            prompt = construct_prompt(
                example["sentence_text"], example["drug1_text"], example["drug2_text"]
            )

            predicted_label = get_hf_llm_prediction(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            if predicted_label != "none":  # Only write if it's a predicted interaction
                outf.write(
                    f"{example['sid']}|{example['e1_id']}|{example['e2_id']}|{predicted_label}\n"
                )

            if args.api_request_delay > 0:  # Renamed from api_request_delay for clarity
                time.sleep(args.api_request_delay)

    print(f"LLM-based DDI predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
