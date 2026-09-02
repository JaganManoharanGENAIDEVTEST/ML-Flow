"""
Chart Recommender Framework - Layer 1 Test Script (No Pytest, No Playwright)
=============================================================================
Purpose: Run the 19 prepared test cases (Chart_Recommender_Test_Dataset.xlsx)
through the REAL chart-recommendation function from your cloned repo, and log
results into MLflow -- ideally reusing the confusion-matrix logger that
already exists in your project.

This is plain Python -- no pytest, no Playwright required. Only needs:
    pip install mlflow pandas openpyxl

BEFORE YOU RUN THIS:
1. Update the import in `call_chart_recommender()` to point to the real
   function in your cloned repo (Step 1 of the guide -- find it via
   Ctrl+Shift+F for "chart_type" / "recommend").
2. Update the import in `log_results_to_mlflow()` to use your project's
   existing confusion-matrix MLflow logger, if one exists (Step 2 of the
   guide). A fallback simple logger is provided in case you can't find one
   or want to compare against it.
3. Update the xlsx path if it's not in the same folder as this script.
"""

import pandas as pd
import mlflow

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TEST_DATASET_PATH = "Chart_Recommender_Test_Dataset.xlsx"
MLFLOW_EXPERIMENT_NAME = "chart_recommender_evaluation"


# ---------------------------------------------------------------------------
# STEP 1: Hook up the real chart recommender function
# ---------------------------------------------------------------------------

def call_chart_recommender(user_question: str, country: str = None) -> str:
    """
    TODO: Replace this with a call to your real chart recommendation function
    from the cloned repo. Example (adjust the import path and arguments to
    match your actual project structure):

        from src.chart_recommender.core import recommend_chart_type
        return recommend_chart_type(user_question, country=country)

    For now this raises an error so you don't accidentally run the script
    without wiring it up first.
    """
    raise NotImplementedError(
        "Wire this up to your real chart recommendation function before running."
    )


# ---------------------------------------------------------------------------
# STEP 2: Log results -- prefer your project's existing logger if one exists
# ---------------------------------------------------------------------------

def log_results_to_mlflow(expected_list, actual_list, question_list):
    """
    Preferred: use your project's existing confusion-matrix MLflow logger.
    Example (adjust the import path to match your actual project structure):

        from src.mlflow_utils.confusion_matrix_logger import log_confusion_matrix
        log_confusion_matrix(expected_list, actual_list)
        return

    Fallback (used below if you haven't wired up the real logger yet): logs
    each test case as its own MLflow run with a simple pass/fail metric, plus
    an overall accuracy metric. This is intentionally simple -- swap it out
    for your project's real logger once you've found it.
    """
    correct = 0
    for question, expected, actual in zip(question_list, expected_list, actual_list):
        with mlflow.start_run(run_name=question[:40]):
            mlflow.log_param("user_question", question)
            mlflow.log_param("expected_chart_type", expected)
            mlflow.log_param("actual_chart_type", actual)
            is_match = (str(expected).strip().lower() == str(actual).strip().lower())
            mlflow.log_metric("match", int(is_match))
            if is_match:
                correct += 1

    accuracy = correct / len(expected_list) if expected_list else 0
    with mlflow.start_run(run_name="OVERALL_SUMMARY"):
        mlflow.log_metric("overall_accuracy", accuracy)
        mlflow.log_metric("total_cases", len(expected_list))
        mlflow.log_metric("correct_cases", correct)

    print(f"\nOverall accuracy: {accuracy:.2%} ({correct}/{len(expected_list)})")


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def main():
    df = pd.read_excel(
        TEST_DATASET_PATH,
        sheet_name="Chart Recommender Test Data",
        header=3,
    )
    df = df.dropna(subset=["Family Chart"])  # drop any blank trailing rows

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    expected_list = []
    actual_list = []
    question_list = []
    results_summary = []

    for _, row in df.iterrows():
        question = row["User Question"]
        expected_chart_type = row["Chart Type"]
        country = row["Country Selection"]

        try:
            actual_chart_type = call_chart_recommender(question, country=country)
        except NotImplementedError:
            raise  # stop immediately -- you haven't wired up the function yet
        except Exception as e:
            actual_chart_type = f"ERROR: {e}"

        expected_list.append(expected_chart_type)
        actual_list.append(actual_chart_type)
        question_list.append(question)

        match = str(expected_chart_type).strip().lower() == str(actual_chart_type).strip().lower()
        results_summary.append({
            "question": question,
            "expected": expected_chart_type,
            "actual": actual_chart_type,
            "match": match,
        })

    log_results_to_mlflow(expected_list, actual_list, question_list)

    summary_df = pd.DataFrame(results_summary)
    print("\n=== Test Run Summary ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("chart_recommender_test_summary.csv", index=False)
    print("\nSaved summary to chart_recommender_test_summary.csv")
    print("Open the MLflow Tracking UI to review full details per test case.")


if __name__ == "__main__":
    main()
