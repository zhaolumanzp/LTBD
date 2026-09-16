import json
import os
from typing import Dict, List, Optional

from datasets import Dataset


# ============================================================
# Cleaned Alpaca
# ============================================================

DEFAULT_ALPACA_PATH = (
    
    "/path/to/AlpacaDataCleaned/alpaca_data_cleaned.json"
)


def load_cleaned_alpaca(
    path: str = DEFAULT_ALPACA_PATH,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load the cleaned Alpaca dataset.

    Expected original format:

    [
        {
            "instruction": "...",
            "input": "...",
            "output": "..."
        },
        ...
    ]

    The returned dataset keeps:
        instruction
        data
        output

    `input` is normalized to `data`.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cleaned Alpaca dataset not found:\n{path}"
        )

    print("=" * 80)
    print("Loading Cleaned Alpaca")
    print(f"Path: {path}")
    print("=" * 80)

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(
            "Expected the Alpaca JSON file to contain a list of samples."
        )

    samples: List[Dict] = []

    for item in raw_data:

        instruction = item.get("instruction", "")

        # Cleaned Alpaca normally uses "input".
        # Some local versions may already use "data".
        if "data" in item:
            data = item.get("data", "")
        else:
            data = item.get("input", "")

        output = item.get("output", "")

        if instruction is None:
            instruction = ""

        if data is None:
            data = ""

        if output is None:
            output = ""

        samples.append(
            {
                "instruction": str(instruction),
                "data": str(data),
                "output": str(output),
            }
        )

    if max_samples is not None:
        samples = samples[:max_samples]

    dataset = Dataset.from_list(samples)

    print(f"Loaded samples: {len(dataset)}")

    if len(dataset) > 0:
        print("\nFirst sample:")
        print(dataset[0])

    print("=" * 80)

    return dataset


def get_cleaned_alpaca_path() -> str:
    """
    Return the default Cleaned Alpaca path.
    """
    return DEFAULT_ALPACA_PATH