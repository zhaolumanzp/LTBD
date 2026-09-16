from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


DELIMITER_TOKENS = [
    "<INST_BEGIN>",
    "<INST_END>",
    "<DATA_BEGIN>",
    "<DATA_END>",
]


@dataclass(frozen=True)
class ChatSample:
    instruction: str  # trusted task instruction
    data: str         # untrusted external data
    response: str     # assistant target


# ============================================================
# Model family
# ============================================================

def _get_model_family(tokenizer: Any) -> str:
    """
    Infer the model family from tokenizer.name_or_path.

    Supported:
      - Llama3 / Llama3.1
      - Qwen2.5
      - Falcon3
    """
    name = str(
        getattr(tokenizer, "name_or_path", "")
    ).lower()

    if "qwen" in name:
        return "qwen"

    if "falcon" in name:
        return "falcon"

    if "llama" in name:
        return "llama"

    raise ValueError(
        f"Unsupported model family. tokenizer.name_or_path={name}"
    )


# ============================================================
# Message construction
# ============================================================

def build_messages(
    instruction: str,
    data: str,
    tokenizer: Any,
) -> List[Dict[str, str]]:
    """
    Build model-specific logical messages.

    Llama3 / Llama3.1:
        system = instruction
        user   = data

    Qwen2.5 / Falcon3:
        user  = instruction
        input = data

    For Qwen/Falcon, "input" is only an internal role consumed by
    our modified chat_template. It is merged into the native user block.
    """

    family = _get_model_family(tokenizer)

    instruction = str(instruction)
    data = str(data)

    # --------------------------------------------------------
    # Llama3 / Llama3.1
    # --------------------------------------------------------
    if family == "llama":
        return [
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": data,
            },
        ]

    # --------------------------------------------------------
    # Qwen2.5 / Falcon3
    # --------------------------------------------------------
    if family in {"qwen", "falcon"}:
        messages = [
            {
                "role": "user",
                "content": instruction,
            }
        ]

        # Only add the synthetic input role when data is non-empty.
        if data != "":
            messages.append(
                {
                    "role": "input",
                    "content": data,
                }
            )

        return messages

    raise ValueError(
        f"Unsupported model family: {family}"
    )


def build_plain_messages(
    instruction: str,
    data: str,
    tokenizer: Any,
) -> List[Dict[str, str]]:
    """
    Backward-compatible helper used by plain/base-model prompting.

    Message-role mapping remains identical to build_messages().
    Whether delimiters are inserted is controlled by
    add_delimiter_tokens in apply_chat_template().
    """
    return build_messages(
        instruction,
        data,
        tokenizer,
    )


# ============================================================
# Delimiter token registration
# ============================================================

def register_delimiter_tokens(tokenizer: Any) -> List[int]:
    """
    Ensure the four delimiter tokens exist in the tokenizer vocab.

    Returns their token ids in DELIMITER_TOKENS order.
    """

    existing = set(
        getattr(
            tokenizer,
            "additional_special_tokens",
            [],
        )
        or []
    )

    to_add = [
        token
        for token in DELIMITER_TOKENS
        if token not in existing
    ]

    if to_add:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens":
                    list(existing) + to_add
            }
        )

    token_ids = [
        int(tokenizer.convert_tokens_to_ids(token))
        for token in DELIMITER_TOKENS
    ]

    return token_ids


# ============================================================
# Chat template check
# ============================================================

def _require_chat_template(tokenizer: Any) -> None:
    """
    We rely on tokenizer.apply_chat_template for model-specific
    native chat delimiters.
    """

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Tokenizer lacks apply_chat_template; "
            "need a chat/instruct tokenizer."
        )

    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "Tokenizer has no chat_template."
        )


# ============================================================
# SFT tokenization
# ============================================================

def tokenize_chat_for_sft(
    tokenizer: Any,
    instruction: str,
    data: str,
    response: str,
    *,
    max_length: int,
) -> Dict[str, Any]:
    """
    Builds input_ids / attention_mask / labels for SFT.

    Only the assistant response contributes to the CE loss.

    Structure:

        prompt_ids:
            model-native chat prompt
            + our INST/DATA delimiter structure
            + assistant generation header

        response_ids:
            assistant target

        labels:
            prompt tokens   -> -100
            response tokens -> original token ids
    """

    _require_chat_template(tokenizer)

    # --------------------------------------------------------
    # Build model-specific messages
    # --------------------------------------------------------

    messages = build_messages(
        instruction,
        data,
        tokenizer,
    )

    # --------------------------------------------------------
    # Render prompt with delimiters
    # --------------------------------------------------------

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_delimiter_tokens=True,
    )

    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    ).input_ids

    # --------------------------------------------------------
    # Tokenize response
    # --------------------------------------------------------

    resp_ids = tokenizer(
        response,
        add_special_tokens=False,
    ).input_ids

    # --------------------------------------------------------
    # Add EOS
    # --------------------------------------------------------

    eos_id = getattr(
        tokenizer,
        "eos_token_id",
        None,
    )

    if (
        eos_id is not None
        and (
            len(resp_ids) == 0
            or resp_ids[-1] != eos_id
        )
    ):
        resp_ids = resp_ids + [eos_id]

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    input_ids = (
        prompt_ids
        + resp_ids
    )[:max_length]

    attention_mask = [
        1
    ] * len(input_ids)

    # --------------------------------------------------------
    # Labels
    #
    # prompt -> ignore
    # response -> train
    # --------------------------------------------------------

    prompt_len = min(
        len(prompt_ids),
        len(input_ids),
    )

    labels = (
        [-100] * prompt_len
        + input_ids[prompt_len:]
    )

    labels = labels[:len(input_ids)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# Render prompt only
# ============================================================

def render_prompt_only(
    tokenizer: Any,
    instruction: str,
    data: str,
    *,
    add_delimiter_tokens: bool = True,
) -> str:
    """
    Render prompt only, without assistant response.

    Uses exactly the same model-specific role mapping as training.
    """

    _require_chat_template(tokenizer)

    messages = build_messages(
        instruction,
        data,
        tokenizer,
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_delimiter_tokens=add_delimiter_tokens,
    )