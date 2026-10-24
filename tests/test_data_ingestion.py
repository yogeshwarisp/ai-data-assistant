import pytest

from src.data_ingestion import IngestionError, make_sql_names, prepare_dataset


def test_basic_profile_and_types():
    csv = b"Order ID,Order Date,Region,Sales\n1,2024-01-05,North,10.5\n2,2024-02-06,South,20\n"
    prepared = prepare_dataset(csv, "x.csv")
    types = {c.original: c.logical_type for c in prepared.profile.columns}
    assert types == {"Order ID": "integer", "Order Date": "datetime", "Region": "text", "Sales": "float"}
    assert prepared.profile.rows == 2
    # ID-like columns are not offered as measures
    assert prepared.profile.numeric_columns == ["Sales"]


def test_sql_names_are_safe_and_unique():
    names = make_sql_names(["Unit Price", "unit-price", "2024 Sales", "Order", ""])
    assert names[0] == "unit_price"
    assert names[1] == "unit_price_2"
    assert names[2] == "col_2024_sales"
    assert names[3] == "order_col"
    assert names[4].startswith("column_")


def test_currency_strings_become_numbers():
    csv = b'Item,Price\na,"$1,200.50"\nb,$30\n'
    prepared = prepare_dataset(csv, "x.csv")
    assert prepared.df["Price"].tolist() == [1200.5, 30.0]


def test_semicolon_delimiter_and_latin1():
    csv = "Name;Amount\nJos\xe9;5\nAna;7\n".encode("latin-1")
    prepared = prepare_dataset(csv, "x.csv")
    assert list(prepared.df.columns) == ["Name", "Amount"]
    assert prepared.df["Name"].iloc[0] == "Jos\xe9"


def test_missing_and_duplicates_counted():
    csv = b"a,b\n1,x\n1,x\n2,\n"
    profile = prepare_dataset(csv, "x.csv").profile
    assert profile.duplicate_rows == 1
    assert profile.total_missing == 1


def test_leading_zero_codes_stay_text():
    csv = b"zip,n\n00123,1\n04567,2\n"
    prepared = prepare_dataset(csv, "x.csv")
    assert prepared.df["zip"].tolist() == ["00123", "04567"]


@pytest.mark.parametrize("raw", [b"", b"   \n  ", b"a,b\n"])
def test_empty_files_raise(raw):
    with pytest.raises(IngestionError):
        prepare_dataset(raw, "x.csv")


def test_sql_frame_has_iso_dates_and_sql_names():
    prepared = prepare_dataset(b"Order Date,Sales\n2024-01-05,1\n", "x.csv")
    frame = prepared.sql_frame()
    assert list(frame.columns) == ["order_date", "sales"]
    assert frame["order_date"].iloc[0] == "2024-01-05"
