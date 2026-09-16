import os
import json
import argparse
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from data_sources import load_cleaned_alpaca


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Self-label Cleaned Alpaca using the original LLM."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the original HuggingFace model.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Generation batch size.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 means greedy decoding.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output instead of resuming.",
    )

    return parser.parse_args()


# ============================================================
# Prompt construction
# ============================================================

def build_user_content(instruction, data):

    instruction = "" if instruction is None else str(instruction)
    data = "" if data is None else str(data)

    instruction = instruction.strip()
    data = data.strip()

    if data:
        return (
            f"{instruction}\n\n"
            f"{data}"
        )

    return instruction


def build_chat_prompt(
    tokenizer,
    instruction,
    data,
):
    """
    Build the original-model prompt.

    IMPORTANT:
    No DefensiveTokens.
    No injection.
    No StruQ delimiter.
    No custom [INST]/[DATA]/[RESP].

    The tokenizer's native chat template is used.
    """

    user_content = build_user_content(
        instruction,
        data,
    )

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    return inputs


# ============================================================
# Generation
# ============================================================

@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    instructions,
    data_list,
    max_new_tokens,
    temperature,
    top_p,
):

    input_ids_list = []

    for instruction, data in zip(
        instructions,
        data_list,
    ):

        input_ids = build_chat_prompt(
            tokenizer,
            instruction,
            data,
        )

        input_ids_list.append(
            input_ids[0]
        )

    # --------------------------------------------------------
    # Left-pad all prompts to the same length
    # --------------------------------------------------------

    max_length = max(
        input_ids.shape[-1]
        for input_ids in input_ids_list
    )

    batch_input_ids = []
    batch_attention_mask = []

    for input_ids in input_ids_list:

        pad_length = (
            max_length
            - input_ids.shape[-1]
        )

        if pad_length > 0:

            padding = torch.full(
                (pad_length,),
                tokenizer.pad_token_id,
                dtype=input_ids.dtype,
            )

            input_ids = torch.cat(
                [
                    padding,
                    input_ids,
                ],
                dim=0,
            )

            attention_mask = torch.cat(
                [
                    torch.zeros(
                        pad_length,
                        dtype=torch.long,
                    ),
                    torch.ones(
                        input_ids.shape[-1]
                        - pad_length,
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )

        else:

            attention_mask = torch.ones(
                input_ids.shape[-1],
                dtype=torch.long,
            )

        batch_input_ids.append(
            input_ids
        )

        batch_attention_mask.append(
            attention_mask
        )

    input_ids = torch.stack(
        batch_input_ids
    ).to(model.device)

    attention_mask = torch.stack(
        batch_attention_mask
    ).to(model.device)

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    if temperature == 0:
        do_sample = False
    else:
        do_sample = True

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Only decode newly generated tokens.
    generated_ids = outputs[
        :,
        max_length:
    ]

    responses = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return [
        response.strip()
        for response in responses
    ]


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "=" * 80)
    print("DefensiveToken Self-Labeling")
    print("=" * 80)

    print(f"Model:          {args.model}")
    print(f"Output:         {args.output}")
    print(f"Max samples:    {args.max_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature:    {args.temperature}")
    print(f"Top-p:          {args.top_p}")
    print(f"Seed:           {args.seed}")
    print("=" * 80 + "\n")

    # --------------------------------------------------------
    # 1. Tokenizer
    # --------------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # 2. Original model
    # --------------------------------------------------------

    print("Loading original model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    # Make absolutely sure that this stage does not train.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # 3. Dataset
    # --------------------------------------------------------

    dataset = load_cleaned_alpaca(
        max_samples=args.max_samples,
    )

    print(
        f"Self-labeling {len(dataset)} samples..."
    )

    # --------------------------------------------------------
    # 4. Output directory
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        os.path.abspath(args.output)
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 5. Resume support
    # --------------------------------------------------------

    start_index = 0

    if (
        os.path.exists(args.output)
        and not args.overwrite
    ):

        with open(
            args.output,
            "r",
            encoding="utf-8",
        ) as f:

            start_index = sum(
                1 for line in f
                if line.strip()
            )

        if start_index > 0:

            print(
                f"Existing output detected: "
                f"{start_index} samples."
            )

            print(
                "Resuming from that position."
            )

    mode = "w" if args.overwrite else "a"

    # --------------------------------------------------------
    # 6. Self-label
    # --------------------------------------------------------

    # Total number of batches over the full dataset.
    total_batches = (
        len(dataset)
        + args.batch_size
        - 1
    ) // args.batch_size

    # Number of full batches already represented by
    # the existing output.
    initial_batches = (
        start_index
        // args.batch_size
    )

    with open(
        args.output,
        mode,
        encoding="utf-8",
    ) as fout:

        for batch_start in tqdm(
            range(
                start_index,
                len(dataset),
                args.batch_size,
            ),
            initial=initial_batches,
            total=total_batches,
            desc="Self-labeling",
        ):

            batch_end = min(
                batch_start + args.batch_size,
                len(dataset),
            )

            # Do not use dataset[batch_start:batch_end].
            batch_samples = [
                dataset[i]
                for i in range(
                    batch_start,
                    batch_end,
                )
            ]

            instructions = [
                sample["instruction"]
                for sample in batch_samples
            ]

            data_list = [
                sample["data"]
                for sample in batch_samples
            ]

            try:

                responses = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    instructions=instructions,
                    data_list=data_list,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

            except Exception as e:

                print(
                    f"\nGeneration failed at batch "
                    f"{batch_start}-{batch_end - 1}: "
                    f"{repr(e)}"
                )

                # Do not silently produce a wrong label.
                raise

            for sample, response in zip(
                batch_samples,
                responses,
            ):

                result = {
                    "instruction": sample["instruction"],
                    "data": sample["data"],
                    "self_response": response,
                }

                fout.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            fout.flush()

    print("\n" + "=" * 80)
    print("Self-labeling finished.")
    print(f"Output: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()