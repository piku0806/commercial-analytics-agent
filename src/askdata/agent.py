"""Question -> semantic query -> compiled SQL -> safety check -> execute -> answer."""
from __future__ import annotations

from datetime import date

import mlflow
from mlflow.entities import SpanType

from askdata import planner, semantic


def _fmt(value, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "usd":
        return f"${float(value):,.0f}"
    if kind == "pct":
        return f"{float(value) * 100:.1f}%"
    return f"{int(value):,}"


def _label(v) -> str:
    return v.strftime("%b %Y") if isinstance(v, date) else str(v)


def summarize(q: semantic.SemanticQuery, cols: list[str], rows: list[tuple]) -> str:
    lay = semantic.layer()
    m = q.metrics[0]
    label, kind = lay["metrics"][m]["label"], lay["metrics"][m]["format"]
    when = (f" from {q.time_range.start:%b %d, %Y} to {q.time_range.end:%b %d, %Y}" if q.time_range else "")
    where = "".join(f" for {', '.join(f.values)}" for f in q.filters)
    if not rows:
        return f"No data{where}{when}."
    mi = cols.index(m)
    if not q.group_by:
        parts = [f"{lay['metrics'][x]['label']}: {_fmt(rows[0][cols.index(x)], lay['metrics'][x]['format'])}" for x in q.metrics]
        return f"{'; '.join(parts)}{where}{when}."
    dim = q.group_by[0]
    dim_label = dim.replace("_", " ")
    if lay["dimensions"][dim].get("time") and (not q.order_by or q.order_by.field == dim):
        first, last = rows[0], rows[-1]
        change = (float(last[mi]) / float(first[mi]) - 1) * 100 if first[mi] else 0
        return (f"{label}{where} by {dim_label}{when}: {_fmt(first[mi], kind)} in {_label(first[0])} to "
                f"{_fmt(last[mi], kind)} in {_label(last[0])} ({change:+.0f}%).")
    if q.order_by and q.limit == 1:
        word = "highest" if q.order_by.direction == "desc" else "lowest"
        return f"{_label(rows[0][0])} had the {word} {label.lower()}{where}{when}: {_fmt(rows[0][mi], kind)}."
    top = ", ".join(f"{_label(r[0])} ({_fmt(r[mi], kind)})" for r in rows[:5])
    return f"{label}{where} by {dim_label}{when}: {top}{'…' if len(rows) > 5 else ''}."


@mlflow.trace(name="ask_data", span_type=SpanType.AGENT)
def ask(question: str) -> dict:
    with mlflow.start_span("plan", span_type=SpanType.LLM) as span:
        span.set_inputs({"question": question})
        try:
            sq = planner.plan(question)
        except planner.Refusal as r:
            span.set_outputs({"refused": str(r)})
            return {"status": "refused", "answer": str(r), "semantic_query": None, "sql": None, "columns": [], "rows": []}
        except (semantic.SemanticError, ValueError) as e:
            span.set_outputs({"error": str(e)})
            return {"status": "error", "answer": f"I couldn't build a valid query: {e}", "semantic_query": None,
                    "sql": None, "columns": [], "rows": []}
        span.set_outputs({"semantic_query": sq.model_dump(mode="json")})
    sql = semantic.compile_sql(sq)
    with mlflow.start_span("execute_sql", span_type=SpanType.TOOL) as span:
        span.set_inputs({"sql": sql})
        cols, rows = semantic.execute(sql)
        span.set_outputs({"columns": cols, "rows": [list(map(str, r)) for r in rows]})
    return {"status": "answered", "answer": summarize(sq, cols, rows), "semantic_query": sq.model_dump(mode="json"),
            "sql": sql, "columns": cols, "rows": rows}
