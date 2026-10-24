from src.data_ingestion import prepare_dataset
from src.local_analysis import find_column, local_answer

CSV = b"Region,Product,Sales,Quantity\nN,A,10,1\nS,B,20,2\nN,A,5,3\n"


def setup():
    p = prepare_dataset(CSV, "x.csv")
    return p.df, p.profile


def test_find_column_prefers_longest_match():
    assert find_column("sum of unit price", ["Price", "Unit Price"]) == "Unit Price"


def test_row_count():
    df, profile = setup()
    assert "3 rows" in local_answer("How many rows are there?", df, profile)[0]


def test_total_sales():
    df, profile = setup()
    assert "35.00" in local_answer("what is the total sales", df, profile)[0]


def test_average_quantity():
    df, profile = setup()
    assert "2.00" in local_answer("average quantity", df, profile)[0]


def test_summary_is_not_mistaken_for_sum():
    df, profile = setup()
    assert "Dataset Summary" in local_answer("summarize the data", df, profile)[0]


def test_non_numeric_column_is_rejected():
    df, profile = setup()
    assert "not a numeric column" in local_answer("total region", df, profile)[0]


def test_unknown_question_gives_help():
    df, profile = setup()
    assert "couldn't interpret" in local_answer("tell me a joke", df, profile)[0]
