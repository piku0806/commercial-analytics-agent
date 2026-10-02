"""Turn a question into a SemanticQuery.

* LLM planner: given the semantic layer catalog, the model returns SemanticQuery JSON. Validation
  errors are fed back once so it can correct itself.
* Rule planner (offline): synonym and pattern matching over the same catalog. It's deterministic,
  which makes it a useful baseline in evals.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta

from askdata.semantic import SemanticError, SemanticQuery, dimension_values, layer, validate


class Refusal(Exception):
    """The question asks for something the semantic layer does not define."""


def as_of() -> date:
    return date.fromisoformat(str(layer()["as_of"]))


def _quarter(year: int, q: int) -> tuple[date, date]:
    start = date(year, 3 * (q - 1) + 1, 1)
    end = date(year + (q == 4), (3 * q) % 12 + 1, 1) - timedelta(days=1)
    return start, end


MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}


def _time_range(q: str) -> tuple[date, date] | None:
    today = as_of()
    if m := re.search(r"\bq([1-4])\s*(20\d\d)\b", q):
        return _quarter(int(m.group(2)), int(m.group(1)))
    if m := re.search(r"\b(" + "|".join(MONTHS) + r")\s+(20\d\d)\b", q):
        mo, yr = MONTHS[m.group(1)], int(m.group(2))
        return date(yr, mo, 1), (date(yr + (mo == 12), mo % 12 + 1, 1) - timedelta(days=1))
    if "last quarter" in q or "previous quarter" in q:
        cq = (today.month - 1) // 3 + 1
        return _quarter(today.year - (cq == 1), 4 if cq == 1 else cq - 1)
    if re.search(r"\b(this year|ytd|year to date)\b", q):
        return date(today.year, 1, 1), today
    if re.search(r"\b(last 12 months|past 12 months|past year|trailing twelve months)\b", q):
        return today - timedelta(days=364), today
    if "last year" in q:
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)
    if m := re.search(r"\b(20\d\d)\b", q):
        y = int(m.group(1))
        return date(y, 1, 1), date(y, 12, 31)
    return None


def rule_plan(question: str) -> SemanticQuery:
    lay, q = layer(), question.lower()
    undefined = [t for t in lay["undefined_terms"] if re.search(rf"\b{re.escape(t)}\b", q)]
    if undefined:
        raise Refusal(f"'{undefined[0]}' isn't defined in the semantic layer, so I won't estimate it. "
                      f"Defined metrics: {', '.join(m['label'] for m in lay['metrics'].values())}.")

    # Metrics: longest synonym first, without reusing matched text ("new prescriptions" != "prescriptions")
    remaining, metrics = q, []
    for syn, name in sorted(((s, n) for n, m in lay["metrics"].items() for s in m["synonyms"]), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(syn)}\b", remaining):
            metrics.append(name)
            remaining = re.sub(rf"\b{re.escape(syn)}\b", " ", remaining)

    # Group-by: a dimension word introduced by by/per/each/across/which/what/top N, or a time word like "monthly"
    group_by = []
    for syn, name in sorted(((s, n) for n, d in lay["dimensions"].items() for s in d["synonyms"]), key=lambda x: -len(x[0])):
        if name in group_by:
            continue
        intro = rf"\b(by|per|each|across|which|what|top\s+\d+|bottom\s+\d+)\s+(the\s+)?{re.escape(syn)}\b"
        bare_time = lay["dimensions"][name].get("time") and syn in ("monthly", "quarterly", "yearly", "annual",
                                                                      "annually", "trend", "over time")
        if re.search(intro, q) or (bare_time and re.search(rf"\b{re.escape(syn)}\b", q)):
            group_by.append(name)

    # Filters: any known dimension value named in the question
    filters = []
    for dim, values in dimension_values().items():
        hits = [v for v in values if re.search(rf"\b{re.escape(v.lower())}\b", q)]
        if hits and dim not in group_by:
            filters.append({"dimension": dim, "op": "in", "values": hits})

    if not metrics:
        raise Refusal("I couldn't match that to a defined metric. Try one of: "
                      + ", ".join(m["label"] for m in lay["metrics"].values()) + ".")

    order_by, limit = None, 100
    if m := re.search(r"\b(top|bottom)\s+(\d+)\b", q):
        order_by, limit = {"field": metrics[0], "direction": "desc" if m.group(1) == "top" else "asc"}, int(m.group(2))
    elif re.search(r"\b(highest|most|best|largest|biggest|leading)\b", q) and group_by:
        order_by, limit = {"field": metrics[0], "direction": "desc"}, 1
    elif re.search(r"\b(lowest|least|worst|smallest)\b", q) and group_by:
        order_by, limit = {"field": metrics[0], "direction": "asc"}, 1

    tr = _time_range(q)
    return SemanticQuery(metrics=metrics, group_by=group_by, filters=filters, order_by=order_by, limit=limit,
                         time_range={"start": tr[0], "end": tr[1]} if tr else None)


SYSTEM = """You translate business questions into a JSON semantic query over a governed semantic layer.
Use ONLY the metrics, dimensions and dimension values in the catalog. Never write SQL.
If the question asks for something not in the catalog (e.g. profit, cost, forecast), return {"refuse": "<reason>"}.
Today is {as_of}. Time ranges are inclusive ISO dates.
Schema: {"metrics": [str], "group_by": [str], "filters": [{"dimension": str, "op": "in"|"!=", "values": [str]}],
"time_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} | null, "order_by": {"field": str, "direction": "asc"|"desc"} | null,
"limit": int}
Return ONLY JSON."""


def llm_plan(question: str) -> SemanticQuery:
    from openai import AzureOpenAI, OpenAI

    p = os.getenv("LLM_PROVIDER", "openai")
    if p == "databricks":
        client, model = OpenAI(base_url=os.environ["DATABRICKS_HOST"].rstrip("/") + "/serving-endpoints",
                               api_key=os.environ["DATABRICKS_TOKEN"]), os.getenv("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")
    elif p == "azure":
        client, model = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                    api_version="2024-10-21"), os.environ["AZURE_OPENAI_DEPLOYMENT"]
    else:
        client, model = OpenAI(), os.getenv("LLM_MODEL", "gpt-4.1-mini")
    lay = layer()
    catalog = {"metrics": {k: {"label": v["label"], "synonyms": v["synonyms"]} for k, v in lay["metrics"].items()},
               "dimensions": {k: v["synonyms"] for k, v in lay["dimensions"].items()},
               "dimension_values": dimension_values()}
    messages = [{"role": "system", "content": SYSTEM.replace("{as_of}", str(lay["as_of"]))},
                {"role": "user", "content": f"Catalog: {json.dumps(catalog)}\n\nQuestion: {question}"}]
    for attempt in range(2):
        text = client.chat.completions.create(model=model, temperature=0, messages=messages).choices[0].message.content
        data = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        if "refuse" in data:
            raise Refusal(data["refuse"])
        try:
            sq = SemanticQuery(**data)
            validate(sq)
            return sq
        except (SemanticError, ValueError) as e:
            if attempt == 1:
                raise
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That query is invalid: {e}. Return a corrected JSON query."}]
    raise SemanticError("planner failed")


def plan(question: str) -> SemanticQuery:
    return rule_plan(question) if os.getenv("LLM_PROVIDER", "mock").lower() == "mock" else llm_plan(question)
