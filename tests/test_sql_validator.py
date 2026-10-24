import pytest

from src.sql_validator import SQLValidationError, validate_sql


def test_accepts_simple_select_and_adds_limit():
    sql = validate_sql('SELECT "region", SUM("sales") FROM "data" GROUP BY "region"', max_rows=50)
    assert sql.endswith("LIMIT 50")


def test_keeps_existing_limit():
    sql = validate_sql('SELECT * FROM data LIMIT 3')
    assert sql.count("LIMIT") == 1


def test_accepts_cte():
    sql = validate_sql('WITH t AS (SELECT region, SUM(sales) s FROM data GROUP BY region) SELECT * FROM t')
    assert sql.startswith("WITH")


def test_trailing_semicolon_is_removed():
    assert ";" not in validate_sql("SELECT 1 FROM data;")


@pytest.mark.parametrize(
    "bad",
    [
        "DROP TABLE data",
        "DELETE FROM data",
        "UPDATE data SET sales = 0",
        "INSERT INTO data VALUES (1)",
        "PRAGMA table_info(data)",
        "SELECT * FROM data; DROP TABLE data",
        "SELECT 1 -- comment",
        "SELECT /* x */ 1 FROM data",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM other_table",
        'SELECT * FROM "other_table"',
        "ATTACH DATABASE 'x.db' AS x",
        "",
        "SELECT 'unterminated FROM data",
    ],
)
def test_rejects_unsafe_sql(bad):
    with pytest.raises(SQLValidationError):
        validate_sql(bad)


def test_keywords_inside_strings_and_identifiers_are_fine():
    validate_sql("SELECT * FROM data WHERE product = 'delete'")
    validate_sql('SELECT "update_flag", "Drop Date" FROM data')


def test_unknown_quoted_column_is_rejected_when_columns_given():
    cols = ["region", "sales"]
    with pytest.raises(SQLValidationError):
        validate_sql('SELECT SUM("revenue") FROM "data"', allowed_columns=cols)


def test_known_columns_and_aliases_pass_when_columns_given():
    cols = ["region", "sales"]
    validate_sql(
        'SELECT "region", SUM("sales") AS "total" FROM "data" GROUP BY "region" ORDER BY "total" DESC',
        allowed_columns=cols,
    )
    validate_sql('SELECT "region" AS r FROM "data" ORDER BY "r"', allowed_columns=cols)
