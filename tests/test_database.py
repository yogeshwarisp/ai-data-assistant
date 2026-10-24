import pytest

from src.data_ingestion import prepare_dataset
from src.database import QueryExecutionError, build_connection, describe_schema, run_query


def make():
    return prepare_dataset(b"Region,Sales\nN,10\nS,20\nN,5\n", "x.csv")


def test_query_runs():
    conn = build_connection(make())
    df = run_query(conn, 'SELECT "region", SUM("sales") AS s FROM "data" GROUP BY "region" ORDER BY s DESC')
    assert df["region"].tolist() == ["S", "N"]


@pytest.mark.parametrize(
    "sql",
    [
        'DELETE FROM "data"',
        'DROP TABLE "data"',
        "INSERT INTO data(region) VALUES ('x')",
        "SELECT * FROM sqlite_master",
    ],
)
def test_database_itself_is_read_only(sql):
    conn = build_connection(make())
    with pytest.raises(Exception):
        conn.execute(sql)


def test_timeout_aborts_runaway_query():
    conn = build_connection(make())
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c"
    with pytest.raises(QueryExecutionError):
        run_query(conn, runaway, timeout_seconds=1)


def test_schema_description_mentions_columns():
    text = describe_schema(make())
    assert '"region"' in text and '"sales"' in text and "range 5 to 20" in text
