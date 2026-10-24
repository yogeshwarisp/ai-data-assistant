"""CSV ingestion: decode, clean, detect types, profile, and make SQL-safe names."""

from __future__ import annotations

import csv
import io
import re
import warnings
from dataclasses import dataclass, field

import pandas as pd

from src.config import get_settings

TABLE_NAME = "data"

# Words that would need quoting in SQLite; we avoid them as column names.
_RESERVED = {
    "select", "from", "where", "group", "order", "by", "having", "limit",
    "table", "index", "key", "default", "check", "primary", "foreign",
    "references", "union", "all", "distinct", "case", "when", "then", "else",
    "end", "and", "or", "not", "null", "in", "is", "like", "between", "join",
    "on", "as", "insert", "update", "delete", "drop", "create", "alter",
    "values", "set", "desc", "asc", "exists", "transaction", "match",
}


class IngestionError(Exception):
    """Raised when an uploaded file cannot be turned into a usable dataset."""


@dataclass
class ColumnProfile:
    original: str
    sql_name: str
    logical_type: str  # integer | float | datetime | boolean | text
    missing: int
    distinct: int


@dataclass
class DatasetProfile:
    source_name: str
    rows: int
    n_columns: int
    duplicate_rows: int
    total_missing: int
    columns: list[ColumnProfile] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)

    @property
    def sql_name_map(self) -> dict[str, str]:
        return {c.original: c.sql_name for c in self.columns}


@dataclass
class PreparedDataset:
    df: pd.DataFrame  # original (human friendly) column names
    profile: DatasetProfile
    table_name: str = TABLE_NAME

    def sql_frame(self) -> pd.DataFrame:
        """Return a copy that is safe to load into SQLite."""
        out = self.df.copy()
        for col in out.columns:
            series = out[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                has_time = (series.dropna().dt.normalize() != series.dropna()).any()
                fmt = "%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d"
                out[col] = series.dt.strftime(fmt)
            elif pd.api.types.is_bool_dtype(series):
                out[col] = series.astype("int64")
        return out.rename(columns=self.profile.sql_name_map)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _is_text(series: pd.Series) -> bool:
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _clean_column_names(columns) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(columns, start=1):
        name = str(raw).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)
    return names


def make_sql_names(columns: list[str]) -> list[str]:
    """Lowercase snake_case, unique, not starting with a digit or reserved."""
    result: list[str] = []
    used: set[str] = set()
    for i, original in enumerate(columns, start=1):
        name = re.sub(r"[^a-z0-9]+", "_", str(original).lower()).strip("_")
        if not name:
            name = f"column_{i}"
        if name[0].isdigit():
            name = f"col_{name}"
        if name in _RESERVED:
            name = f"{name}_col"
        base, n = name, 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        result.append(name)
    return result


def _try_numeric(series: pd.Series) -> pd.Series | None:
    """Convert strings like '1,234.50', '$12', '15%' to numbers if nearly all fit."""
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return None
    # Keep codes like 00123 as text.
    if non_null.str.match(r"^0\d+$").any():
        return None
    cleaned = non_null.str.replace(r"[,$€£₹%\s]", "", regex=True)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    if parsed.notna().mean() < 0.95:
        return None
    full = series.astype(str).str.strip().str.replace(r"[,$€£₹%\s]", "", regex=True)
    return pd.to_numeric(full.where(series.notna()), errors="coerce")


_DATE_LIKE = r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"


def _try_datetime(series: pd.Series) -> pd.Series | None:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None
    if non_null.str.match(_DATE_LIKE).mean() < 0.9:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return None
    if parsed.notna().sum() / max(series.notna().sum(), 1) < 0.9:
        return None
    return parsed


def _logical_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "text"


def _looks_like_id(name: str) -> bool:
    return bool(re.search(r"(^|[\s_\-])id$", name.strip().lower()))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def prepare_dataset(csv_data: bytes, filename: str = "data.csv") -> PreparedDataset:
    """Turn raw CSV bytes into a cleaned DataFrame plus a profile."""
    settings = get_settings()

    if not csv_data or not csv_data.strip():
        raise IngestionError("The file is empty.")
    if len(csv_data) > settings.max_upload_mb * 1024 * 1024:
        raise IngestionError(f"The file is larger than {settings.max_upload_mb} MB.")

    text = _decode(csv_data)
    delimiter = _sniff_delimiter(text)

    try:
        df = pd.read_csv(
            io.StringIO(text), sep=delimiter, skipinitialspace=True, dtype=str
        )
    except pd.errors.EmptyDataError as exc:
        raise IngestionError("The file has no readable data.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise IngestionError(f"The file is not a valid CSV ({exc}).") from exc

    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.empty or len(df.columns) == 0:
        raise IngestionError("The CSV has no data rows.")
    if len(df) > settings.max_rows:
        raise IngestionError(f"The CSV has more than {settings.max_rows:,} rows.")

    df.columns = _clean_column_names(df.columns)

    # Clean text cells and upgrade column types where it is safe to do so.
    for col in df.columns:
        series = df[col]
        if not _is_text(series):
            continue
        series = series.astype("object").where(series.isna(), series.astype(str).str.strip())
        series = series.replace("", pd.NA)
        converted = _try_numeric(series)
        if converted is None:
            converted = _try_datetime(series)
        if converted is None:
            converted = series
        df[col] = converted

    sql_names = make_sql_names(list(df.columns))
    columns = [
        ColumnProfile(
            original=col,
            sql_name=sql,
            logical_type=_logical_type(df[col]),
            missing=int(df[col].isna().sum()),
            distinct=int(df[col].nunique(dropna=True)),
        )
        for col, sql in zip(df.columns, sql_names)
    ]

    numeric_columns = [
        c.original
        for c in columns
        if c.logical_type in ("integer", "float") and not _looks_like_id(c.original)
    ]

    profile = DatasetProfile(
        source_name=filename,
        rows=len(df),
        n_columns=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        total_missing=int(df.isna().sum().sum()),
        columns=columns,
        numeric_columns=numeric_columns,
    )
    return PreparedDataset(df=df, profile=profile)
