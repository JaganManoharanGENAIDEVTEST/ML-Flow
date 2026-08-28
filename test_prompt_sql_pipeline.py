"""
MLflow Prompt-to-SQL Evaluation Test Harness
==============================================
Purpose: Run the 19 prepared test cases (MLflow_Prompt_SQL_Test_Dataset.xlsx)
through the actual prompt-to-SQL pipeline, validate against the 8 confirmed
metrics, and log every result into MLflow so it's visible in the Tracking UI.

BEFORE YOU RUN THIS:
1. Update the import in `call_pipeline()` to point to your real prompt-to-SQL
   function (Step 2 of the guide -- find it in the existing codebase).
2. Update the Snowflake connection details in `get_snowflake_connection()`.
3. Update ALLOWED_TABLES / ALLOWED_COLUMNS to match your real approved schema.
4. Update LATENCY_SLA_SECONDS and NUMERIC_TOLERANCE_PCT to your team's agreed
   thresholds (these are currently placeholders -- see Section 9 open items).
5. Update the xlsx path if it's not in the same folder as this script.

Run it with:
    python test_prompt_sql_pipeline.py
"""

import re
import time
import math
import pandas as pd
import mlflow

# ---------------------------------------------------------------------------
# CONFIG -- adjust these to match your actual project setup
# ---------------------------------------------------------------------------

TEST_DATASET_PATH = "MLflow_Prompt_SQL_Test_Dataset.xlsx"
MLFLOW_EXPERIMENT_NAME = "prompt_to_sql_evaluation"

# TODO: confirm exact threshold with your team (Section 9, open item)
LATENCY_SLA_SECONDS = 3.0

# TODO: confirm exact tolerance with your team (Section 9, open item)
NUMERIC_TOLERANCE_PCT = 0.5  # allow +/- 0.5% difference on aggregate values

# TODO: replace with your real approved table/column names
ALLOWED_TABLES = {"products", "sales"}
ALLOWED_COLUMNS = {
    "product_id", "product_name", "category", "manufacturer", "unit_price",
    "sale_id", "sale_date", "quantity", "sales_amount", "region", "customer_type",
}

# Keywords that should never appear in a generated SQL for this read-only
# analytics use case -- flags unsafe/destructive queries.
UNSAFE_KEYWORDS = {"DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE", "GRANT"}


# ---------------------------------------------------------------------------
# STEP 2: Hook up the real pipeline
# ---------------------------------------------------------------------------

def call_pipeline(prompt: str) -> str:
    """
    TODO: Replace this with a call to your actual prompt-to-SQL function.
    Example (adjust the import path to match your project structure):

        from src.pipeline.nl_to_sql import generate_sql
        return generate_sql(prompt)

    For now this is a placeholder so the script runs end-to-end for review.
    """
    raise NotImplementedError(
        "Wire this up to your real prompt-to-SQL function before running."
    )


def get_snowflake_connection():
    """
    TODO: Replace with your real Snowflake connection.
    Example:

        import snowflake.connector
        return snowflake.connector.connect(
            account="your_account",
            user="your_user",
            password="your_password",   # better: use env vars / secrets manager
            warehouse="your_warehouse",
            database="your_database",
            schema="your_schema",
        )
    """
    raise NotImplementedError("Wire this up to your real Snowflake connection.")


def run_query(conn, sql: str) -> pd.DataFrame:
    """Executes a SQL string against Snowflake and returns a DataFrame."""
    return pd.read_sql(sql, conn)


# ---------------------------------------------------------------------------
# STEP 5: One check function per metric
# ---------------------------------------------------------------------------

def check_error_free(conn, sql: str):
    """Returns (passed: bool, detail: str)"""
    try:
        run_query(conn, sql)
        return True, "Query executed without error."
    except Exception as e:
        return False, f"Execution error: {e}"


def check_strict_schema(sql: str):
    """Flags any table/column-like token not in the approved allow-list.
    This is a lightweight heuristic, not a full SQL parser -- refine as needed."""
    tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", sql.lower()))
    sql_keywords = {
        "select", "from", "where", "join", "on", "group", "by", "order",
        "having", "as", "and", "or", "not", "in", "limit", "sum", "avg",
        "count", "distinct", "over", "partition", "date_trunc", "dateadd",
        "year", "current_date", "round", "desc", "asc", "null", "is",
    }
    suspicious = tokens - sql_keywords - ALLOWED_TABLES - ALLOWED_COLUMNS
    suspicious = {t for t in suspicious if not t.isdigit() and len(t) > 2}
    if suspicious:
        return False, f"References unexpected identifiers: {suspicious}"
    return True, "Only approved tables/columns referenced."


def check_safety(sql: str, prompt: str):
    """Flags destructive SQL keywords or signs of a followed injection attempt."""
    sql_upper = sql.upper()
    found_unsafe = [kw for kw in UNSAFE_KEYWORDS if kw in sql_upper]
    if found_unsafe:
        return False, f"Unsafe keywords found: {found_unsafe}"
    injection_markers = ["ignore previous instructions", "ignore your instructions"]
    if any(m in prompt.lower() for m in injection_markers):
        # If the prompt was an injection attempt, the SQL should NOT look like
        # a normal successful query -- flag for manual review either way.
        return False, "Prompt was an injection attempt -- manually verify system refused correctly."
    return True, "No unsafe keywords or injection compliance detected."


def check_numeric_tolerance(actual_df: pd.DataFrame, expected_df: pd.DataFrame, tolerance_pct: float = NUMERIC_TOLERANCE_PCT):
    """Compares numeric columns within a tolerance instead of exact match."""
    try:
        numeric_cols = actual_df.select_dtypes(include="number").columns
        for col in numeric_cols:
            if col not in expected_df.columns:
                continue
            a = actual_df[col].sum()
            e = expected_df[col].sum()
            if e == 0:
                if a != 0:
                    return False, f"Column {col}: expected 0, got {a}"
                continue
            pct_diff = abs(a - e) / abs(e) * 100
            if pct_diff > tolerance_pct:
                return False, f"Column {col}: {pct_diff:.2f}% difference exceeds {tolerance_pct}% tolerance"
        return True, "All numeric columns within tolerance."
    except Exception as e:
        return False, f"Could not compare numeric columns: {e}"


def check_equivalence(actual_df: pd.DataFrame, expected_df: pd.DataFrame):
    """Checks if two result sets are equivalent regardless of row/column order."""
    try:
        a_sorted = actual_df.sort_index(axis=1).sort_values(by=list(actual_df.columns)).reset_index(drop=True)
        e_sorted = expected_df.sort_index(axis=1).sort_values(by=list(expected_df.columns)).reset_index(drop=True)
        if a_sorted.equals(e_sorted):
            return True, "Result sets are equivalent."
        return False, "Result sets differ after sorting -- manual review needed."
    except Exception as e:
        return False, f"Could not compare result sets: {e}"


def check_latency(elapsed_seconds: float):
    if elapsed_seconds <= LATENCY_SLA_SECONDS:
        return True, f"{elapsed_seconds:.2f}s within {LATENCY_SLA_SECONDS}s SLA."
    return False, f"{elapsed_seconds:.2f}s exceeds {LATENCY_SLA_SECONDS}s SLA."


# ---------------------------------------------------------------------------
# STEP 3 + 4 + 6: Load dataset, run pipeline, log to MLflow
# ---------------------------------------------------------------------------

def main():
    df = pd.read_excel(TEST_DATASET_PATH, sheet_name="Prompt-SQL Test Dataset", header=4)
    df = df.dropna(subset=["ID"])  # drop any blank trailing rows

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    conn = get_snowflake_connection()

    results_summary = []

    for _, row in df.iterrows():
        test_id = row["ID"]
        prompt = row["Natural Language Prompt"]
        expected_sql = row["Expected Snowflake SQL"]

        with mlflow.start_run(run_name=test_id):
            mlflow.log_param("test_id", test_id)
            mlflow.log_param("category", row["Category"])
            mlflow.log_param("prompt", prompt)
            mlflow.log_param("complexity", row["Complexity"])

            metric_results = {}

            # --- Call the real pipeline and measure latency ---
            start = time.time()
            try:
                actual_sql = call_pipeline(prompt)
                pipeline_error = None
            except Exception as e:
                actual_sql = ""
                pipeline_error = str(e)
            elapsed = time.time() - start

            mlflow.log_text(actual_sql or "N/A", f"{test_id}_generated_sql.sql")
            mlflow.log_text(str(expected_sql), f"{test_id}_expected_sql.sql")

            # --- Latency SLA ---
            passed, detail = check_latency(elapsed)
            metric_results["latency_sla"] = passed
            mlflow.log_metric("latency_sla_pass", int(passed))
            mlflow.log_metric("latency_seconds", elapsed)

            if pipeline_error:
                mlflow.log_param("pipeline_error", pipeline_error)
                metric_results["error_free"] = False
            else:
                # --- Error Free ---
                passed, detail = check_error_free(conn, actual_sql)
                metric_results["error_free"] = passed
                mlflow.log_metric("error_free_pass", int(passed))

                # --- Strict Schema ---
                passed, detail = check_strict_schema(actual_sql)
                metric_results["strict_schema"] = passed
                mlflow.log_metric("strict_schema_pass", int(passed))

                # --- Safety ---
                passed, detail = check_safety(actual_sql, prompt)
                metric_results["safety"] = passed
                mlflow.log_metric("safety_pass", int(passed))

                # --- Correctness / Equivalence / Numeric Tolerance ---
                # Only meaningful for cases with a real expected query (skip
                # for edge cases like P008/P010/P011 where expected_sql is a
                # comment, not a runnable query).
                if isinstance(expected_sql, str) and expected_sql.strip().upper().startswith("SELECT"):
                    try:
                        actual_df = run_query(conn, actual_sql)
                        expected_df = run_query(conn, expected_sql)

                        passed, detail = check_equivalence(actual_df, expected_df)
                        metric_results["equivalence"] = passed
                        mlflow.log_metric("equivalence_pass", int(passed))

                        passed, detail = check_numeric_tolerance(actual_df, expected_df)
                        metric_results["numeric_tolerance"] = passed
                        mlflow.log_metric("numeric_tolerance_pass", int(passed))

                        # Correctness: treat as pass if equivalence passed
                        metric_results["correctness"] = metric_results["equivalence"]
                        mlflow.log_metric("correctness_pass", int(metric_results["correctness"]))
                    except Exception as e:
                        mlflow.log_param("comparison_error", str(e))

                # Relevance to Query is left for manual review -- log the
                # prompt + generated SQL together (already done above) so a
                # human can quickly judge it in the MLflow UI.

            results_summary.append({"test_id": test_id, **metric_results})

    conn.close()

    summary_df = pd.DataFrame(results_summary)
    print("\n=== Test Run Summary ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("test_run_summary.csv", index=False)
    print("\nSaved summary to test_run_summary.csv")
    print("Open the MLflow Tracking UI to review full details per test case.")


if __name__ == "__main__":
    main()
