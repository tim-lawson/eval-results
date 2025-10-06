# Thanks to Gemini 2.5 Pro

import json
import os
import re
from pathlib import Path

import pandas as pd


def parse_eval_results(root_dir: str) -> pd.DataFrame:
    """Parse math-evaluation-harness JSON results into a DataFrame."""
    assert os.path.isdir(root_dir), f"Directory '{root_dir}' does not exist"

    # Find all JSON files within any 'math_eval*' subdirectory.
    # The pattern matches: output/{model_type}/{revision}/math_eval{SEED}/{task}/*.json
    files = list(Path(root_dir).glob("*/*/math_eval*/*/*.json"))

    if not files:
        return pd.DataFrame()

    results = []
    for filepath in files:
        try:
            # Extract metadata from filepath
            parts = filepath.parts
            model_type = parts[-5]
            revision = parts[-4]
            math_eval_dir = parts[-3]
            task_name = parts[-2]

            step = int(revision.replace("step", ""))

            seed_match = re.search(r"\d+$", math_eval_dir)
            seed = int(seed_match.group()) if seed_match else 0

            # Parse JSON and extract accuracy
            with filepath.open("r") as f:
                content = json.load(f)

            acc = content.get("acc")
            if acc is not None:
                results.append(
                    {
                        "model_type": model_type,
                        "step": step,
                        "seed": seed,
                        "task": task_name,
                        "accuracy": acc,
                    }
                )
        except (IndexError, json.JSONDecodeError):
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.pivot_table(
        index=["model_type", "step", "seed"], columns="task", values="accuracy"
    ).reset_index()
    df = df.sort_values(["model_type", "step", "seed"])
    cols = ["model_type", "step"] + [
        c for c in df.columns if c not in ["model_type", "step"]
    ]
    df = df[cols]  # Move columns to the front

    columns = {}
    for c in df.columns:
        columns[c] = c.replace("_", "").replace(",", "")
    return df.rename(columns=columns)


if __name__ == "__main__":
    results = parse_eval_results("output")
    if not results.empty:
        results.to_csv("math_eval_results.csv", index=False)
        print(f"Results saved. Shape: {results.shape}")
        print(results.head())
    else:
        print("No results found.")
