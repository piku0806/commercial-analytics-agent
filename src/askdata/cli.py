"""Ask a question about the commercial data.

    python -m askdata "Which region had the highest net sales in 2025?"
    python -m askdata "Monthly NRx for Xelvora this year" --dialect snowflake
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

import mlflow  # noqa: E402

from askdata import semantic  # noqa: E402
from askdata.agent import ask  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(prog="askdata")
    p.add_argument("question", nargs="+")
    p.add_argument("--dialect", help="also print the SQL for snowflake, databricks or bigquery")
    a = p.parse_args()
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("commercial-analytics-agent")
    r = ask(" ".join(a.question))
    print(r["answer"])
    if r["status"] != "answered":
        return
    print("\nSemantic query:", json.dumps(r["semantic_query"]))
    print("\nSQL (DuckDB):\n" + r["sql"])
    if a.dialect:
        print(f"\nSQL ({a.dialect}):\n" + semantic.transpile(r["sql"], a.dialect))
    print("\n" + " | ".join(r["columns"]))
    for row in r["rows"][:12]:
        print(" | ".join(str(v) for v in row))


if __name__ == "__main__":
    main()
