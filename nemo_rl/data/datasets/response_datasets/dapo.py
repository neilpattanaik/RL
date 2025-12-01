from typing import Any

from datasets import Dataset, load_dataset

from nemo_rl.data.datasets.aime_val import get_formatted_aime_val_ds
from nemo_rl.data.interfaces import TaskDataSpec

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
        "task_name": "math",
    }


def prepare_dapo_dataset(seed: int, aime_year: str) -> dict[str, Dataset | None]:
    """Load and split the DAPO dataset into train and test sets."""
    # Load the original dataset for training
    train_ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    train_ds = train_ds.shuffle(seed=seed)
    train_formatted = train_ds.map(
        format_dapo_math, remove_columns=train_ds.column_names
    )
    val_formatted = get_formatted_aime_val_ds(aime_year)

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class DapoMathDataset:
    def __init__(self, seed: int = 42, aime_year: str = "24_25") -> None:
        """Initialize the DAPO Math dataset with AIME val.

        Args:
            seed: Random seed for reproducible splitting
            aime_year: AIME year to use for validation set. Options are '24', '25', and '24_25'.
        """
        self.formatted_ds = prepare_dapo_dataset(seed=seed, aime_year=aime_year)

        self.task_spec = TaskDataSpec(
            task_name="DapoMath",
        )
