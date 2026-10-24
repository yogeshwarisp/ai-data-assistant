"""AI Data Assistant - Streamlit app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.data_ingestion import IngestionError, prepare_dataset  # noqa: E402
from src.database import build_connection  # noqa: E402
from src.local_analysis import local_answer  # noqa: E402
from src.text_to_sql import answer_question  # noqa: E402

logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Data Assistant", page_icon="📊", layout="wide")

settings = get_settings()
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_sales.csv"


# --------------------------------------------------
# Small helpers
# --------------------------------------------------
def show_df(data, **kwargs) -> None:
    """st.dataframe that works before and after Streamlit's width= API change."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:
        st.dataframe(data, use_container_width=True, **kwargs)


def show_chart(fig, **kwargs) -> None:
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


@st.cache_data(show_spinner="Reading and profiling your CSV...")
def load_dataset(csv_bytes: bytes, filename: str):
    return prepare_dataset(csv_bytes, filename)


def auto_chart(result: pd.DataFrame):
    """Make a simple bar chart for 'label + number' style query results."""
    if result is None or not (2 <= len(result) <= 50) or result.shape[1] != 2:
        return None
    label, value = result.columns
    if not pd.api.types.is_numeric_dtype(result[value]):
        return None
    return px.bar(result, x=label, y=value, title=f"{value} by {label}")


def render_message(message: dict, key: str) -> None:
    st.markdown(message["content"])
    if message.get("sql"):
        with st.expander("Generated SQL"):
            st.code(message["sql"], language="sql")
    if message.get("table") is not None:
        show_df(message["table"])
    if message.get("figure") is not None:
        show_chart(message["figure"], key=f"fig_{key}")


# --------------------------------------------------
# Header and data source
# --------------------------------------------------
st.title("📊 AI Data Assistant")
st.write("Explore your data, create charts, and ask questions in plain English.")
st.caption(f"Maximum upload size: {settings.max_upload_mb} MB")

st.sidebar.header("Data Source")
use_sample = st.sidebar.checkbox("Use sample sales dataset", value=True)
uploaded_file = st.sidebar.file_uploader("Upload your own CSV file", type=["csv"])

csv_data: bytes | None = None
csv_filename: str | None = None

if uploaded_file is not None:
    csv_data = uploaded_file.getvalue()
    csv_filename = uploaded_file.name
    st.sidebar.success("Using uploaded CSV.")
elif use_sample and SAMPLE_CSV.exists():
    csv_data = SAMPLE_CSV.read_bytes()
    csv_filename = SAMPLE_CSV.name
    st.sidebar.success("Using sample sales dataset.")
elif use_sample:
    st.sidebar.error(
        f"Sample CSV not found at {SAMPLE_CSV}. "
        "Run `python scripts/generate_sample_data.py` to create it."
    )

if csv_data is None:
    st.info("Select the sample dataset or upload a CSV file using the sidebar.")
    st.stop()

st.sidebar.divider()
if settings.ai_enabled:
    st.sidebar.success(f"AI mode: OpenAI ({settings.openai_model})")
else:
    st.sidebar.info(
        "Local mode: set OPENAI_API_KEY and OPENAI_MODEL in `.env` to enable AI questions."
    )

# --------------------------------------------------
# Process the dataset
# --------------------------------------------------
try:
    prepared = load_dataset(csv_data, csv_filename)
except IngestionError as exc:
    st.error(f"Could not process the CSV: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    logger.exception("Unexpected error while loading dataset")
    st.error(f"An unexpected error occurred while loading the dataset: {exc}")
    st.stop()

df = prepared.df
profile = prepared.profile

# Reset chat + DB connection whenever the dataset changes.
signature = (csv_filename, hashlib.md5(csv_data).hexdigest())
if st.session_state.get("dataset_signature") != signature:
    st.session_state.dataset_signature = signature
    st.session_state.chat_history = []
    st.session_state.pop("db_conn", None)

st.success(f"Loaded dataset: {profile.source_name}")

# --------------------------------------------------
# Dashboard
# --------------------------------------------------
st.header("📈 Dataset Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows", f"{profile.rows:,}")
c2.metric("Total Columns", profile.n_columns)
c3.metric("Duplicate Rows", profile.duplicate_rows)
c4.metric("Missing Values", profile.total_missing)

with st.expander("Data Preview", expanded=True):
    show_df(df.head(settings.max_result_rows))

with st.expander("Column Profile"):
    show_df(
        pd.DataFrame(
            [
                {
                    "Original Column": c.original,
                    "SQL Column": c.sql_name,
                    "Data Type": c.logical_type,
                    "Missing Values": c.missing,
                    "Distinct Values": c.distinct,
                }
                for c in profile.columns
            ]
        )
    )

with st.expander("Summary Statistics"):
    show_df(df.describe(include="all").transpose().astype(str))

# --------------------------------------------------
# Interactive charts
# --------------------------------------------------
st.subheader("📊 Interactive Charts")

numeric_columns = profile.numeric_columns
if numeric_columns:
    left, right = st.columns(2)
    with left:
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"],
            key="chart_type",
        )
    with right:
        y_axis = st.selectbox("Numeric Column", numeric_columns, key="chart_y")

    fig = None
    if chart_type == "Histogram":
        fig = px.histogram(df, x=y_axis, title=f"Distribution of {y_axis}")
    else:
        x_options = [c for c in df.columns if c != y_axis]
        if not x_options:
            st.info("A second column is required for this chart.")
        else:
            x_axis = st.selectbox("Horizontal Axis", x_options, key="chart_x")
            if chart_type == "Bar Chart":
                # Aggregate categories so bars are totals, not thousands of stacked rows.
                is_category = not pd.api.types.is_numeric_dtype(df[x_axis]) and not (
                    pd.api.types.is_datetime64_any_dtype(df[x_axis])
                )
                if is_category and df[x_axis].nunique() <= 100:
                    plot_df = df.groupby(x_axis, dropna=True)[y_axis].sum().reset_index()
                    title = f"Total {y_axis} by {x_axis}"
                else:
                    plot_df, title = df, f"{y_axis} by {x_axis}"
                fig = px.bar(plot_df, x=x_axis, y=y_axis, title=title)
            elif chart_type == "Line Chart":
                fig = px.line(df.sort_values(x_axis), x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")

    if fig is not None:
        show_chart(fig, key="dashboard_chart")
else:
    st.info("No numeric columns were detected for charting.")

# --------------------------------------------------
# Export
# --------------------------------------------------
st.subheader("Export Data")
st.download_button(
    label="Download Processed CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="processed_data.csv",
    mime="text/csv",
)

# --------------------------------------------------
# Chat with your data
# --------------------------------------------------
st.divider()
st.header("🤖 Chat with Your Data")

if settings.ai_enabled:
    st.write(
        "Ask anything about the dataset. The AI writes a SQL query, it is validated and run "
        "read-only on a local SQLite copy, and you see both the SQL and the result."
    )
    st.caption(
        "Privacy note: the question, column names, and a few example values per column are "
        "sent to OpenAI. Full rows are never sent."
    )
else:
    st.write(
        "Ask common questions such as *total sales by region*, *average quantity*, "
        "*missing values*, or *top products*. Add an OpenAI key to unlock free-form questions."
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for i, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        render_message(message, key=str(i))

prompt = st.chat_input("Ask a question about your data...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply = {"role": "assistant", "content": "", "sql": None, "table": None, "figure": None}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            note = ""
            if settings.ai_enabled:
                try:
                    if "db_conn" not in st.session_state:
                        st.session_state.db_conn = build_connection(prepared)
                    qa = answer_question(
                        prompt, prepared, settings, conn=st.session_state.db_conn
                    )
                except Exception as exc:  # noqa: BLE001 - never crash the UI
                    logger.exception("AI pipeline failed")
                    qa = None
                    note = f"_AI mode failed ({exc}); used local analysis instead._\n\n"
                else:
                    if qa.ok:
                        shown = qa.result.head(settings.max_result_rows)
                        reply.update(
                            content=f"Here is the result ({len(qa.result):,} row(s)).",
                            sql=qa.sql,
                            table=shown,
                            figure=auto_chart(qa.result),
                        )
                    elif qa.error and qa.error.startswith("Not answerable"):
                        reply["content"] = (
                            f"I can't answer that from this dataset. {qa.error.split(': ', 1)[-1]}"
                        )
                    else:
                        note = (
                            f"_The AI query could not be completed ({qa.error}); "
                            "trying local analysis._\n\n"
                        )

            if not reply["content"]:
                text, figure = local_answer(prompt, df, profile)
                reply.update(content=note + text, figure=figure)

        render_message(reply, key=str(len(st.session_state.chat_history)))

    st.session_state.chat_history.append(reply)

if st.session_state.chat_history and st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
