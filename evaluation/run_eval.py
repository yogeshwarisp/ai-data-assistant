"""Evaluate the text-to-SQL pipeline against gold queries.

Usage (from the project root):
    python -m evaluation.run_eval --check-gold     # no API key needed: validates the question set
    python -m evaluation.run_eval                  # runs the LLM (needs OPENAI_API_KEY + OPENAI_MODEL)
    python -m evaluation.run_eval --limit 10       # quick run on the first 10 questions

Metrics are *measured* from the run - nothing is hard-coded. Execution accuracy means the
generated query returns the same values as the gold query (column names are ignored,
floats are rounded to 2 decimals, row order only matters when a question is marked ordered).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, get_settings
from src.data_ingestion import PreparedDataset, prepare_dataset
from src.database import QueryExecutionError, build_connection, run_query
from src.sql_validator import SQLValidationError, validate_sql
from src.text_to_sql import answer_question

QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_sales.csv"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sample() -> PreparedDataset:
    return prepare_dataset(SAMPLE_CSV.read_bytes(), SAMPLE_CSV.name)


def _normalize(df: pd.DataFrame, ordered: bool) -> list[tuple]:
    rows = []
    for row in df.itertuples(index=False, name=None):
        rows.append(tuple(round(v, 2) if isinstance(v, float) else v for v in row))
    return rows if ordered else sorted(rows, key=lambda r: tuple(str(v) for v in r))


def results_match(generated: pd.DataFrame, gold: pd.DataFrame, ordered: bool = False) -> bool:
    if generated.shape != gold.shape:
        return False
    return _normalize(generated, ordered) == _normalize(gold, ordered)


def run_gold_query(conn, sql: str, prepared: PreparedDataset, max_rows: int = 10_000) -> pd.DataFrame:
    columns = [c.sql_name for c in prepared.profile.columns]
    return run_query(conn, validate_sql(sql, max_rows=max_rows, allowed_columns=columns))


def check_gold() -> list[dict]:
    """Make sure every gold query passes validation and executes."""
    prepared = load_sample()
    conn = build_connection(prepared)
    problems = []
    for q in load_questions():
        if not q["answerable"]:
            continue
        try:
            df = run_gold_query(conn, q["gold_sql"], prepared)
            if df.empty:
                problems.append({"id": q["id"], "problem": "gold query returned no rows"})
        except (SQLValidationError, QueryExecutionError) as exc:
            problems.append({"id": q["id"], "problem": str(exc)})
    return problems


def evaluate(limit: int | None = None, client=None) -> dict:
    settings = get_settings()
    if client is None and not settings.ai_enabled:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL in .env to run the LLM evaluation.")

    prepared = load_sample()
    conn = build_connection(prepared)
    questions = load_questions()[: limit or None]

    records = []
    for q in questions:
        start = time.perf_counter()
        qa = answer_question(q["question"], prepared, settings, client=client, conn=conn)
        latency = time.perf_counter() - start

        record = {
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "answerable": q["answerable"],
            "generated_sql": qa.sql,
            "valid_and_executed": qa.ok,
            "attempts": qa.attempts,
            "latency_seconds": round(latency, 3),
            "error": qa.error,
        }
        if q["answerable"]:
            correct = False
            if qa.ok:
                gold = run_gold_query(conn, q["gold_sql"], prepared)
                correct = results_match(qa.result, gold, q["ordered"])
            record["correct"] = correct
        else:
            record["correct"] = (not qa.ok) and bool(qa.error and "Not answerable" in qa.error)
        records.append(record)

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    def rate(hits: int, total: int) -> float | None:
        return round(hits / total, 4) if total else None

    by_category: dict[str, dict] = {}
    for r in answerable:
        c = by_category.setdefault(r["category"], {"total": 0, "correct": 0})
        c["total"] += 1
        c["correct"] += int(r["correct"])

    summary = {
        "model": settings.openai_model,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "questions_run": len(records),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "execution_accuracy": rate(sum(r["correct"] for r in answerable), len(answerable)),
        "valid_sql_rate": rate(sum(r["valid_and_executed"] for r in answerable), len(answerable)),
        "correct_refusal_rate": rate(sum(r["correct"] for r in unanswerable), len(unanswerable)),
        "needed_repair_retry": sum(1 for r in records if r["attempts"] > 1),
        "avg_latency_seconds": round(sum(r["latency_seconds"] for r in records) / len(records), 3)
        if records else None,
        "by_category": by_category,
    }
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--check-gold", action="store_true", help="validate gold queries only (no API calls)")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = parser.parse_args()

    if args.check_gold:
        problems = check_gold()
        total = sum(1 for q in load_questions() if q["answerable"])
        if problems:
            print(f"{len(problems)} of {total} gold queries have problems:")
            for p in problems:
                print(f"  - question {p['id']}: {p['problem']}")
            raise SystemExit(1)
        print(f"All {total} gold queries pass validation and execute on the sample data.")
        return

    report = evaluate(limit=args.limit)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
