from typing import Any

from datasets import Dataset, load_dataset, concatenate_datasets

def _format_aime_math(data: dict[str, str | float | int]) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": data["expected_answer"],
            },
        ],
        "task_name": "math",
    }


def _rekey_aime(data: dict[str, Any], input_key: str) -> dict[str, Any]:
    return {
        "problem": data[input_key],
        "expected_answer": data["answer"],
    }

def get_formatted_aime_val_ds(aime_year: str) -> Dataset:
    """Loads formatted dataset for AIME (24, 25, or both), formats, and repeats each entry 16 times."""
    val_ds2024 = None
    val_ds2025 = None

    if "24" in aime_year:
        val_ds2024 = load_dataset("HuggingFaceH4/aime_2024", split="train")
        val_ds2024 = val_ds2024.map(
            _rekey_aime,
            fn_kwargs={"input_key": "problem"},
            remove_columns=val_ds2024.column_names,
        )

    if "25" in aime_year:
        val_ds2025_0 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        val_ds2025_1 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        val_ds2025 = concatenate_datasets([val_ds2025_0, val_ds2025_1])
        val_ds2025 = val_ds2025.map(
            _rekey_aime,
            fn_kwargs={"input_key": "question"},
            remove_columns=val_ds2025.column_names,
        )

    match aime_year:
        case "24":
            val_ds = val_ds2024
        case "25":
            val_ds = val_ds2025
        case "24_25":
            val_ds = concatenate_datasets([val_ds2024, val_ds2025])

    val_formatted = val_ds.map(_format_aime_math, remove_columns=val_ds.column_names)

    val_repeated = []
    for _ in range(16):
        val_repeated.extend(val_formatted)
    val_formatted = val_formatted.from_list(val_repeated)

    return val_formatted
    