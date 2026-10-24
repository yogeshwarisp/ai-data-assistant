# AI Data Assistant

Upload a CSV (or use the built-in sample), explore it in a dashboard, and ask questions in plain English.
In **AI mode** an LLM turns your question into SQL, the SQL is validated and executed **read-only** on a
local SQLite copy of your data, and you see the SQL and the result. Without an API key the app still
works in **local mode** (totals, averages, group-by, missing values, top values, charts).

## Project structure

```
ai-data-assistant/
├── app.py                      # Streamlit UI (entry point)
├── requirements.txt
├── .env.example                # copy to .env and add your key
├── pytest.ini
├── data/
│   └── sample_sales.csv        # 400 generated sample orders
├── src/
│   ├── config.py               # settings from .env
│   ├── data_ingestion.py       # decode, clean, type-detect, profile, SQL-safe names
│   ├── database.py             # in-memory SQLite, read-only, timeout, schema text for the prompt
│   ├── sql_validator.py        # static safety checks on generated SQL
│   ├── text_to_sql.py          # OpenAI Responses API -> validate -> run -> one repair retry
│   └── local_analysis.py       # offline rule-based answers (no API key)
├── evaluation/
│   ├── questions.json          # 35 questions (33 answerable + 2 that should be refused) with gold SQL
│   ├── run_eval.py             # measures execution accuracy, validity, refusals, latency
│   └── results/                # saved eval runs (git-ignored)
├── scripts/
│   └── generate_sample_data.py
└── tests/                      # pytest suite
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Run the app from the project root:

```bash
streamlit run app.py
```

For AI mode, edit `.env` and set **both** `OPENAI_API_KEY` and `OPENAI_MODEL` (any model name your
account can use). Restart Streamlit after changing `.env`.

## Tests and evaluation

```bash
pytest                                  # unit tests (no API key needed)
python -m evaluation.run_eval --check-gold   # checks that all gold queries run (no API key needed)
python -m evaluation.run_eval                # real LLM evaluation, writes evaluation/results/eval_*.json
python -m evaluation.run_eval --limit 10     # quick run
```

The evaluation reports numbers measured from your run only. **Execution accuracy** = the generated
query returns the same values as the gold query (column names ignored, floats rounded to 2 decimals,
row order checked only for questions marked `ordered`). Run it with your own model and quote that result.

## How safety works (two independent layers)

1. **`sql_validator.py`** accepts exactly one `SELECT`/`WITH` statement; rejects comments, multiple
   statements, `INSERT/UPDATE/DELETE/DROP/PRAGMA/ATTACH...`, system tables, unknown tables and unknown
   quoted columns (SQLite would otherwise treat a wrong `"column"` as a text literal); adds a `LIMIT`.
2. **`database.py`** runs on an in-memory copy with `PRAGMA query_only = ON`, a SQLite *authorizer*
   that only allows reads, and a progress-handler timeout for runaway queries.

If a query fails, the error is sent back to the model once for a corrected attempt.

## Privacy

In AI mode the question, column names and up to 5 example values per text column are sent to OpenAI.
Full rows are never sent, and SQL runs locally. Don't use AI mode with confidential data you can't share.

## What was fixed from the original `app.py`

- It imported `src.config` and `src.data_ingestion`, which didn't exist, so it crashed on start-up. Both are now written.
- The old OpenAI path only sent 20 sample rows and could not give exact answers; it now generates SQL that runs on the full data.
- Local analysis matched substrings (`"sum"` inside `"summary"`) and had no group-by; it now uses whole-word matching and supports "total sales by region".
- Added `tabulate` (needed by `to_markdown`), dataset-change reset for chat/DB, Streamlit width API compatibility, unique chart keys, and bar charts that aggregate categories.

## Limitations

Single table only; no joins across files. Dates are detected for `YYYY-MM-DD`-style columns.
Percent strings like `15%` become `15`, not `0.15`.
