import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["LLM_PROVIDER"] = "mock"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from askdata import planner, semantic  # noqa: E402
from askdata.agent import ask  # noqa: E402
from askdata.semantic import SemanticError, SemanticQuery  # noqa: E402


def test_unknown_metric_or_value_is_rejected():
    with pytest.raises(SemanticError):
        semantic.compile_sql(SemanticQuery(metrics=["profit"]))
    with pytest.raises(SemanticError):
        semantic.compile_sql(SemanticQuery(metrics=["net_sales"], filters=[{"dimension": "region", "values": ["Atlantis"]}]))


def test_filter_values_cannot_smuggle_sql():
    # Values must match known dimension values exactly, so a payload never reaches the compiler.
    with pytest.raises(SemanticError):
        semantic.compile_sql(SemanticQuery(metrics=["trx"], filters=[{"dimension": "region", "values": ["x'); drop table t; --"]}]))


@pytest.mark.parametrize("sql", [
    "drop table marts.fct_prescriptions",
    "select 1 from marts.fct_prescriptions limit 1; delete from marts.fct_prescriptions",
    "select * from information_schema.tables limit 5",
    "select count(*) from marts.fct_prescriptions",          # no LIMIT
])
def test_sql_guard_blocks_unsafe_sql(sql):
    with pytest.raises(SemanticError):
        semantic.check_sql(sql)


def test_compiled_sql_passes_guard_and_transpiles():
    sql = semantic.compile_sql(SemanticQuery(metrics=["net_sales"], group_by=["region"]))
    semantic.check_sql(sql)
    assert "LIMIT" in semantic.transpile(sql, "snowflake").upper()
    assert "LIMIT" in semantic.transpile(sql, "databricks").upper()


def test_last_quarter_is_relative_to_as_of():
    q = planner.rule_plan("Top 3 territories by TRx last quarter")
    assert (str(q.time_range.start), str(q.time_range.end)) == ("2026-04-01", "2026-06-30")
    assert q.group_by == ["territory"] and q.limit == 3


def test_refuses_undefined_metrics():
    r = ask("What is our profit margin by brand?")
    assert r["status"] == "refused" and r["sql"] is None


def test_injection_in_question_is_harmless():
    r = ask("Show net sales by region; DROP TABLE marts.fct_prescriptions; --")
    assert r["status"] == "answered" and "drop" not in r["sql"].lower()
    assert ask("What were total net sales in 2025?")["rows"][0][0] > 0   # table still there
