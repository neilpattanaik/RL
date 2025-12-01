from typing import Any

from datasets import Dataset, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec

from nemo_rl.data.datasets.aime_val import get_formatted_aime_val_ds


def format_open_math_reasoning(
    data: dict[str, str | float | int],
) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["question"],
            },
            {
                "role": "assistant",
                "content": data["expected_answer"],
            },
        ],
        "task_name": "math",
    }

def prepare_open_math_reasoning_dataset(
    seed: int,
    aime_year: str,
) -> dict[str, Dataset | None]:
    train_ds = load_dataset("nvidia/Nemotron-RL-math-OpenMathReasoning", split="train")    
    train_ds = train_ds.shuffle(seed=seed)
    train_formatted = train_ds.map(
        format_open_math_reasoning,
        remove_columns=train_ds.column_names,
    )

    val_formatted = get_formatted_aime_val_ds(aime_year)
    
    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class OpenMathReasoningDataset:
    def __init__(
        self,
        seed: int = 42,
        aime_year: str = "24_25",
    ):
        """Initialize the nvidia/Nemotron-RL-math-OpenMathReasoning Math dataset with AIME val.

        Args:
            seed: Random seed for reproducible splitting
            aime_year: AIME year to use for validation set. Options are '24', '25', and '24_25'.
        """        
        self.formatted_ds = prepare_open_math_reasoning_dataset(
            seed=seed, aime_year=aime_year
        )

        self.task_spec = TaskDataSpec(
            task_name="OpenMathReasoning",
        )
