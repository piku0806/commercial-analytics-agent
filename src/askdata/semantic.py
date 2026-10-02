"""Semantic layer: load definitions, validate semantic queries, compile them to SQL, and check that SQL."""
from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import duckdb
import sqlglot
import yaml
from pydantic import BaseModel, Field, field_validator
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[2]
LAYER_FILE = ROOT / "semantic" / "commercial.yml"
WAREHOUSE = Path(os.getenv("ASKDATA_WAREHOUSE", ROOT / "data" / "commercial.duckdb"))


class Filter(BaseModel):
    dimension: str
    op: Literal["=", "!=", "in"] = "in"
    values: list[str]


class TimeRange(BaseModel):
    start: date
    end: date  # inclusive


class OrderBy(BaseModel):
    field: str
    direction: Literal["asc", "desc"] = "desc"


class SemanticQuery(BaseModel):
    metrics: list[str] = Field(min_length=1)
    group_by: list[str] = []
    filters: list[Filter] = []
    time_range: TimeRange | None = None
    order_by: OrderBy | None = None
    limit: int = Field(default=100, ge=1, le=1000)

    @field_validator("metrics", "group_by")
    @classmethod
    def dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


class SemanticError(ValueError):
    pass


@lru_cache
def layer() -> dict:
    return yaml.safe_load(LAYER_FILE.read_text())


@lru_cache
def dimension_values() -> dict[str, list[str]]:
    """Distinct values of each categorical dimension: used for filter validation and by the planner."""
    lay, out = layer(), {}
    with duckdb.connect(str(WAREHOUSE), read_only=True) as con:
        joins = " ".join(f"join {j['table']} {j['alias']} on {j['join_on']}" for j in lay["joins"])
        for name, d in lay["dimensions"].items():
            if not d.get("time"):
                rows = con.execute(f"select distinct {d['expr']} from {lay['base']['table']} {lay['base']['alias']} "
                                   f"{joins} order by 1").fetchall()
                out[name] = [str(r[0]) for r in rows]
    return out


def validate(q: SemanticQuery) -> None:
    lay, vals = layer(), dimension_values()
    bad_m = [m for m in q.metrics if m not in lay["metrics"]]
    bad_d = [d for d in q.group_by + [f.dimension for f in q.filters] if d not in lay["dimensions"]]
    if bad_m:
        raise SemanticError(f"Unknown metric(s) {bad_m}. Available: {sorted(lay['metrics'])}")
    if bad_d:
        raise SemanticError(f"Unknown dimension(s) {bad_d}. Available: {sorted(lay['dimensions'])}")
    for f in q.filters:
        if f.dimension not in vals:
            raise SemanticError(f"Cannot filter on time dimension {f.dimension!r}; use time_range")
        unknown = [v for v in f.values if v not in vals[f.dimension]]
        if unknown:
            raise SemanticError(f"Unknown value(s) {unknown} for {f.dimension}. Valid: {vals[f.dimension]}")
    if q.order_by and q.order_by.field not in q.metrics + q.group_by:
        raise SemanticError(f"order_by field {q.order_by.field!r} must be a selected metric or dimension")
    if q.time_range and q.time_range.start > q.time_range.end:
        raise SemanticError("time_range start is after end")


def _lit(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def compile_sql(q: SemanticQuery) -> str:
    """Deterministic compilation: the model never writes SQL, so it can't invent joins or columns."""
    validate(q)
    lay = layer()
    select = [f"{lay['dimensions'][d]['expr']} as {d}" for d in q.group_by]
    select += [f"{lay['metrics'][m]['expr']} as {m}" for m in q.metrics]
    sql = f"select {', '.join(select)}\nfrom {lay['base']['table']} {lay['base']['alias']}"
    for j in lay["joins"]:
        sql += f"\njoin {j['table']} {j['alias']} on {j['join_on']}"
    where = []
    if q.time_range:
        where.append(f"{lay['time_column']} between date {_lit(q.time_range.start.isoformat())} "
                     f"and date {_lit(q.time_range.end.isoformat())}")
    for f in q.filters:
        col = lay["dimensions"][f.dimension]["expr"]
        vals = ", ".join(_lit(v) for v in f.values)
        where.append(f"{col} {'not in' if f.op == '!=' else 'in'} ({vals})")
    if where:
        sql += "\nwhere " + "\n  and ".join(where)
    if q.group_by:
        sql += "\ngroup by " + ", ".join(str(i + 1) for i in range(len(q.group_by)))
    order = q.order_by
    if order is None and q.group_by:   # time series ascending, rankings largest first
        order = (OrderBy(field=q.group_by[0], direction="asc") if lay["dimensions"][q.group_by[0]].get("time")
                 else OrderBy(field=q.metrics[0], direction="desc"))
    if order:
        sql += f"\norder by {order.field} {order.direction}"
    return sql + f"\nlimit {q.limit}"


def allowed_tables() -> set[str]:
    lay = layer()
    return {lay["base"]["table"], *(j["table"] for j in lay["joins"])}


def check_sql(sql: str) -> None:
    """Defense in depth: even compiled SQL must be a single read-only SELECT over allowlisted tables."""
    statements = sqlglot.parse(sql, read="duckdb")
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise SemanticError("Only a single SELECT statement is allowed")
    stmt = statements[0]
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command)
    if any(stmt.find(t) for t in forbidden):
        raise SemanticError("Write or DDL operations are not allowed")
    tables = {f"{t.db}.{t.name}" if t.db else t.name for t in stmt.find_all(exp.Table)}
    if not tables <= allowed_tables():
        raise SemanticError(f"Query touches non-allowlisted tables: {sorted(tables - allowed_tables())}")
    if not stmt.args.get("limit"):
        raise SemanticError("Query must have a LIMIT")


def transpile(sql: str, dialect: str) -> str:
    """Same query for Snowflake, Databricks or BigQuery."""
    return sqlglot.transpile(sql, read="duckdb", write=dialect, pretty=True)[0]


def execute(sql: str) -> tuple[list[str], list[tuple]]:
    check_sql(sql)
    with duckdb.connect(str(WAREHOUSE), read_only=True) as con:
        cur = con.execute(sql)
        return [d[0] for d in cur.description], cur.fetchall()
