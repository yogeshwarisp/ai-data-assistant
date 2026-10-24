"""Rule-based answers computed directly with pandas (no API key needed)."""

from __future__ import annotations

import re

import pandas as pd

from src.data_ingestion import DatasetProfile

OPERATIONS = {
    "sum": ["total", "sum", "overall"],
    "mean": ["average", "avg", "mean"],
    "max": ["highest", "maximum", "largest", "max"],
    "min": ["lowest", "minimum", "smallest", "min"],
}
LABELS = {"sum": "Total", "mean": "Average", "max": "Maximum", "min": "Minimum"}


def _has_word(text: str, words) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def _has_phrase(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def find_column(question: str, columns) -> str | None:
    """Find a column mentioned by name (longest names are tried first)."""
    q = question.lower()
    q_norm = re.sub(r"[^a-z0-9]+", " ", q)
    for column in sorted(columns, key=lambda c: len(str(c)), reverse=True):
        name = str(column).lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", name).strip()
        if name and name in q:
            return column
        if normalized and re.search(rf"\b{re.escape(normalized)}\b", q_norm):
            return column
    return None


def _default_measure(numeric_columns: list[str]) -> str | None:
    for c in numeric_columns:
        if "sales" in c.lower() or "revenue" in c.lower():
            return c
    return numeric_columns[0] if len(numeric_columns) == 1 else None


def _bar(df: pd.DataFrame, x: str, y: str, title: str):
    import plotly.express as px  # lazy import

    return px.bar(df, x=x, y=y, title=title)


def local_answer(question: str, df: pd.DataFrame, profile: DatasetProfile):
    """Return (markdown_text, plotly_figure_or_None)."""
    q = question.lower().strip()
    numeric_columns = [c for c in profile.numeric_columns if c in df.columns]
    column = find_column(question, list(df.columns))

    if _has_phrase(q, ["how many rows", "row count", "number of rows", "how many records"]):
        return f"The dataset contains **{len(df):,} rows**.", None

    if _has_phrase(q, ["how many columns", "column count", "number of columns"]):
        return f"The dataset contains **{len(df.columns)} columns**.", None

    if _has_word(q, ["missing"]) or "null values" in q:
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            return "There are no missing values in this dataset.", None
        lines = "\n".join(f"- **{n}**: {c}" for n, c in missing.items())
        return f"Missing values by column:\n\n{lines}", None

    if _has_word(q, ["duplicate", "duplicates"]):
        return f"The dataset contains **{int(df.duplicated().sum()):,} duplicate rows**.", None

    if _has_word(q, ["summarize", "summary", "overview"]) or _has_phrase(
        q, ["describe the data", "dataset information"]
    ):
        stats = (
            df[numeric_columns].describe().round(2).to_string()
            if numeric_columns
            else "No numeric columns were found."
        )
        return (
            "### Dataset Summary\n\n"
            f"- **File:** {profile.source_name}\n"
            f"- **Rows:** {len(df):,}\n"
            f"- **Columns:** {len(df.columns)}\n"
            f"- **Duplicate rows:** {int(df.duplicated().sum())}\n"
            f"- **Missing values:** {int(df.isna().sum().sum())}\n\n"
            f"**Numeric statistics**\n\n```text\n{stats}\n```",
            None,
        )

    # Aggregations, optionally grouped: "total sales by region"
    for op, keywords in OPERATIONS.items():
        if not _has_word(q, keywords):
            continue

        target = column if column in numeric_columns else None
        if target is None and column is None:
            target = _default_measure(numeric_columns)
        if target is None and column is not None and column not in numeric_columns:
            return f"`{column}` is not a numeric column.", None
        if target is None:
            return (
                "Please specify a numeric column. Available columns: "
                + ", ".join(f"`{c}`" for c in numeric_columns),
                None,
            )

        group = None
        if re.search(r"\b(by|per|for each|across)\b", q):
            tail = re.split(r"\b(?:by|per|for each|across)\b", q, maxsplit=1)[1]
            group = find_column(tail, [c for c in df.columns if c != target])

        if group is not None:
            result = (
                df.groupby(group, dropna=True)[target]
                .agg(op)
                .sort_values(ascending=False)
                .reset_index()
            )
            title = f"{LABELS[op]} {target} by {group}"
            table = result.head(25).to_markdown(index=False, floatfmt=",.2f")
            fig = None
            if len(result) <= 50:
                try:
                    fig = _bar(result, group, target, title)
                except ImportError:
                    fig = None
            return f"**{title}**\n\n{table}", fig

        value = getattr(df[target], op)()
        return f"**{LABELS[op]} {target}:** {value:,.2f}", None

    # Top values
    if _has_word(q, ["top", "best", "popular"]):
        if column is not None and column not in numeric_columns:
            counts = df[column].value_counts().head(10).to_frame("Count")
            return (
                f"Most frequent values in **{column}**:\n\n" + counts.to_markdown(),
                None,
            )
        if numeric_columns:
            target = column if column in numeric_columns else (
                _default_measure(numeric_columns) or numeric_columns[0]
            )
            result = df.nlargest(10, target)
            return (
                f"Top 10 rows by **{target}**:\n\n" + result.to_markdown(index=False),
                None,
            )

    # Chart requests
    if _has_word(q, ["chart", "plot", "graph", "visualize", "visualise", "histogram"]):
        if not numeric_columns:
            return "No numeric columns are available for charting.", None
        y_axis = column if column in numeric_columns else None
        if y_axis is None:
            y_axis = _default_measure(numeric_columns)
        if y_axis is None:
            return (
                "Please specify a numeric column to chart: "
                + ", ".join(f"`{c}`" for c in numeric_columns),
                None,
            )
        import plotly.express as px  # lazy import

        others = [c for c in df.columns if c != y_axis]
        x_axis = find_column(q, others) or (others[0] if others else None)
        if _has_word(q, ["histogram", "distribution"]) or x_axis is None:
            fig = px.histogram(df, x=y_axis, title=f"Distribution of {y_axis}")
        elif _has_word(q, ["line", "trend"]) or "over time" in q:
            fig = px.line(df.sort_values(x_axis), x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        elif "scatter" in q:
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        else:
            plot_df, title = df, f"{y_axis} by {x_axis}"
            is_category = not pd.api.types.is_numeric_dtype(df[x_axis]) and not (
                pd.api.types.is_datetime64_any_dtype(df[x_axis])
            )
            if is_category and df[x_axis].nunique() <= 100:
                plot_df = df.groupby(x_axis, dropna=True)[y_axis].sum().reset_index()
                title = f"Total {y_axis} by {x_axis}"
            fig = px.bar(plot_df, x=x_axis, y=y_axis, title=title)
        return f"Here is a chart of **{y_axis}**.", fig

    return (
        "I couldn't interpret that question with the built-in analysis. Try asking for "
        "a total or average (e.g. *total sales by region*), missing values, duplicate "
        "rows, a dataset summary, top values, or a chart. For free-form questions, add "
        "an OpenAI API key in `.env` to enable AI mode.",
        None,
    )
