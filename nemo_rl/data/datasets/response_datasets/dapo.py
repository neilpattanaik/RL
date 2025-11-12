from typing import Any

from datasets import Dataset, load_dataset, concatenate_datasets

from nemo_rl.data.interfaces import TaskDataSpec

"""
Copied mostly from deepscaler.py
"""


def format_dapo_math(data: dict[str, str | float | int]) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["prompt"][0]["content"][178:-63],
            },
            {
                "role": "assistant",
                "content": data["reward_model"]["ground_truth"],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def format_aime_math(data: dict[str, str | float | int]) -> dict[str, list[Any] | str]:
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
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def _rekey_aime(data: dict[str, Any], input_key: str) -> dict[str, Any]:
    return {
        "problem": data[input_key],
        "expected_answer": data["answer"],
    }


def prepare_dapo_dataset(seed: int, aime_year: str) -> dict[str, Dataset | None]:
    """Load and split the DAPO dataset into train and test sets."""
    # Load the original dataset for training
    train_ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")

    # Load aime combined dataset for validation
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

    # Shuffle the training dataset with the specified seed
    train_ds = train_ds.shuffle(seed=seed)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(
        format_dapo_math, remove_columns=train_ds.column_names
    )
    val_formatted = val_ds.map(format_aime_math, remove_columns=val_ds.column_names)

    # Compute accuracy 16 times per sample (matching the DeepScaleR evaluation setting)
    val_repeated = []
    for _ in range(16):
        val_repeated.extend(val_formatted)
    val_formatted = val_formatted.from_list(val_repeated)

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class DapoMathDataset:
    def __init__(self, seed: int = 42, aime_year: str = "24_25") -> None:
        """Initialize the DAPO Math dataset with train/test split.

        Args:
            seed: Random seed for reproducible splitting
            aime_year: AIME year to use for validation set. Options are '24', '25', and '24_25'.
        """
        self.formatted_ds = prepare_dapo_dataset(seed=seed, aime_year=aime_year)

        self.task_spec = TaskDataSpec(
            task_name="DapoMath",
        )
