# Thanks to Gemini 2.5 Pro and Claude Sonnet 4

import json
import os
from pathlib import Path

import pandas as pd


def parse_eval_results(root_dir: str) -> pd.DataFrame:
    """Parse lm-evaluation-harness JSON results into a DataFrame."""
    assert os.path.isdir(root_dir), f"Directory '{root_dir}' does not exist"

    # Find all result files, keeping only the latest per directory
    files_by_dir = {}
    for dirpath, _, filenames in os.walk(root_dir):
        files = [
            os.path.join(dirpath, f)
            for f in filenames
            if f.startswith("results_") and f.endswith(".json")
        ]
        if files:
            files_by_dir[dirpath] = max(files, key=os.path.getmtime)

    if not files_by_dir:
        return pd.DataFrame()

    results = []
    for filepath in files_by_dir.values():
        path = Path(filepath)

        # Find step directory and extract metadata
        step_parts = [p for p in path.parts if p.startswith("step")]
        if not step_parts:
            continue

        step_part = step_parts[0]
        step_index = path.parts.index(step_part)
        model_type = path.parts[step_index - 1]
        step = int(step_part.replace("step", ""))

        # Parse JSON and extract metrics
        try:
            with open(filepath) as f:
                data = json.load(f)

            record: dict = {"model_type": model_type, "step": step}

            for task_name, task_results in data.get("results", {}).items():
                if isinstance(task_results, dict):
                    for metric_name, metric_value in task_results.items():
                        if isinstance(metric_value, (int, float)):
                            record[f"{task_name}_{metric_name}"] = metric_value

            results.append(record)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values(["model_type", "step"])
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
        results.to_csv("lm_eval_results.csv", index=False)
        print(f"Results saved. Shape: {results.shape}")
        print(results.head())
    else:
        print("No results found.")
