"""
Generic Automated Test Runner (MLflow-based)
==============================================
Purpose: A reusable framework to run ANY set of test cases (current or
future) through your existing agent-scoring code, and log every result
into MLflow automatically -- no code changes needed to add new test cases.

Design principle:
    - Test cases live in DATA FILES (CSV/xlsx), not in this script.
    - To add future test cases: just add new rows to a dataset file, or
      add a new dataset file and register it in test_suites_config.json.
    - To add future metrics: just add new columns to the dataset -- the
      script logs whatever columns it finds, without needing new code.
    - The ONLY thing you wire up once is call_agent() -- your existing
      working single-prompt logic goes there, unchanged.

Run it with:
    python automated_test_runner.py                     (runs all registered suites)
    python automated_test_runner.py --suite chart_recommender   (runs one suite only)
"""

import argparse
import json
import os
import time
import pandas as pd
import mlflow


# ---------------------------------------------------------------------------
# CONFIG FILE: test_suites_config.json
# ---------------------------------------------------------------------------
# This is where you register test suites -- current and future. Example:
#
# {
#   "suites": [
#     {
#       "name": "chart_recommender",
#       "dataset_path": "Chart_Recommender_Test_Dataset.csv",
#       "prompt_column": "User Question",
#       "expected_column": "Chart Type",
#       "mlflow_experiment": "chart_recommender_evaluation"
#     },
#     {
#       "name": "prompt_to_sql",
#       "dataset_path": "MLflow_Prompt_SQL_Test_Dataset.xlsx",
#       "sheet_name": "Prompt-SQL Test Dataset",
#       "header_row": 4,
#       "prompt_column": "Natural Language Prompt",
#       "expected_column": "Expected Snowflake SQL",
#       "mlflow_experiment": "prompt_to_sql_evaluation"
#     }
#   ]
# }
#
# Adding a THIRD future test type (e.g. a new agent capability) means adding
# one more block to this list -- this script never needs to change.

CONFIG_PATH = "test_suites_config.json"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"{CONFIG_PATH} not found. Create it to register your test suites "
            f"(see the example format in the comments at the top of this script)."
        )
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# STEP: Wire up your EXISTING working single-prompt call here, once.
# ---------------------------------------------------------------------------

def call_agent(prompt: str, **extra_columns):
    """
    TODO (only wire this up ONCE): Replace the body of this function with
    your existing, already-working code that sends a single prompt to the
    agent and gets a response + score back. This is the exact logic you
    already use to validate one prompt's score today -- move it here as-is.

    `extra_columns` will contain any other columns from the dataset row
    (e.g. Country Selection, Complexity) in case your agent call needs them.

    Must return a dict, e.g.:
        {
            "actual_output": "<SQL text or chart type or whatever the agent returned>",
            "score": 0.87,                 # or whatever your current scoring logic produces
            "latency_seconds": 1.42,
            # add any other fields your current code already produces
        }
    """
    raise NotImplementedError(
        "Paste your existing working single-prompt call + scoring logic here."
    )


# ---------------------------------------------------------------------------
# Generic dataset loader -- works for any CSV or xlsx, current or future
# ---------------------------------------------------------------------------

def load_dataset(suite: dict) -> pd.DataFrame:
    path = suite["dataset_path"]
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(
            path,
            sheet_name=suite.get("sheet_name", 0),
            header=suite.get("header_row", 0),
        )
    # Drop fully blank trailing rows using whichever column is the prompt column
    df = df.dropna(subset=[suite["prompt_column"]])
    return df


# ---------------------------------------------------------------------------
# Generic runner -- works for any suite, current or future
# ---------------------------------------------------------------------------

def run_suite(suite: dict):
    print(f"\n=== Running suite: {suite['name']} ===")
    df = load_dataset(suite)
    mlflow.set_experiment(suite.get("mlflow_experiment", suite["name"]))

    prompt_col = suite["prompt_column"]
    expected_col = suite.get("expected_column")

    results_summary = []

    for idx, row in df.iterrows():
        prompt = row[prompt_col]
        expected = row.get(expected_col) if expected_col else None

        # Pass every other column along in case call_agent() needs it
        extra_columns = {
            col: row[col] for col in df.columns if col not in (prompt_col, expected_col)
        }

        run_name = f"{suite['name']}_{idx}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("suite", suite["name"])
            mlflow.log_param("prompt", str(prompt))
            if expected is not None:
                mlflow.log_param("expected", str(expected))

            # Log any extra dataset columns as params automatically --
            # future columns (new metrics, new dimensions) get logged
            # without needing new code here.
            for col, val in extra_columns.items():
                safe_key = str(col).lower().replace(" ", "_")[:50]
                mlflow.log_param(safe_key, str(val)[:250])

            start = time.time()
            try:
                result = call_agent(prompt, **extra_columns)
                error = None
            except NotImplementedError:
                raise  # stop immediately -- call_agent() isn't wired up yet
            except Exception as e:
                result = {}
                error = str(e)
            elapsed = time.time() - start

            mlflow.log_metric("latency_seconds", result.get("latency_seconds", elapsed))

            actual_output = result.get("actual_output", "")
            mlflow.log_text(str(actual_output), f"{run_name}_actual_output.txt")

            if error:
                mlflow.log_param("error", error)

            # Log every score-like field the agent's response contains --
            # this makes the runner future-proof for new metrics your
            # scoring logic might add later, with zero code changes here.
            for key, val in result.items():
                if key in ("actual_output",):
                    continue
                if isinstance(val, (int, float)):
                    mlflow.log_metric(key, val)
                else:
                    mlflow.log_param(key, str(val)[:250])

            match = None
            if expected is not None and actual_output:
                match = str(expected).strip().lower() == str(actual_output).strip().lower()
                mlflow.log_metric("match", int(match))

            results_summary.append({
                "suite": suite["name"],
                "prompt": prompt,
                "expected": expected,
                "actual_output": actual_output,
                "match": match,
                "error": error,
            })

    summary_df = pd.DataFrame(results_summary)
    out_file = f"{suite['name']}_test_summary.csv"
    summary_df.to_csv(out_file, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved summary to {out_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generic MLflow test runner.")
    parser.add_argument("--suite", help="Run only this suite name (optional).", default=None)
    args = parser.parse_args()

    config = load_config()
    suites = config["suites"]

    if args.suite:
        suites = [s for s in suites if s["name"] == args.suite]
        if not suites:
            raise ValueError(f"No suite named '{args.suite}' found in {CONFIG_PATH}")

    for suite in suites:
        run_suite(suite)


if __name__ == "__main__":
    main()
