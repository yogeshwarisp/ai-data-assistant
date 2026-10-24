import pytest

from src.config import Settings
from src.data_ingestion import prepare_dataset
from src.text_to_sql import UnanswerableError, answer_question, extract_sql

SETTINGS = Settings("key", "fake-model", 20, 1000, 200, 5)


class FakeClient:
    """Mimics client.responses.create(...).output_text, returning scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        return type("R", (), {"output_text": reply})()


def prepared():
    return prepare_dataset(b"Region,Sales\nN,10\nS,20\nN,5\n", "x.csv")


def test_extract_sql_from_fence():
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


def test_extract_sql_unanswerable():
    with pytest.raises(UnanswerableError):
        extract_sql("CANNOT_ANSWER: no such column")


def test_happy_path():
    client = FakeClient(['```sql\nSELECT SUM("sales") AS total FROM "data"\n```'])
    qa = answer_question("total?", prepared(), SETTINGS, client=client)
    assert qa.ok and qa.attempts == 1
    assert qa.result["total"].iloc[0] == 35


def test_repair_retry_after_bad_sql():
    client = FakeClient([
        '```sql\nSELECT SUM("revenue") FROM "data"\n```',
        '```sql\nSELECT SUM("sales") FROM "data"\n```',
    ])
    qa = answer_question("total?", prepared(), SETTINGS, client=client)
    assert qa.ok and qa.attempts == 2
    assert "previous query failed" in client.calls[1]["input"]


def test_unsafe_sql_is_never_executed():
    client = FakeClient(["```sql\nDROP TABLE data\n```", "```sql\nDELETE FROM data\n```"])
    qa = answer_question("wipe it", prepared(), SETTINGS, client=client)
    assert not qa.ok and "SELECT" in qa.error


def test_unanswerable_question():
    client = FakeClient(["CANNOT_ANSWER: no profit column"])
    qa = answer_question("profit?", prepared(), SETTINGS, client=client)
    assert not qa.ok and qa.error.startswith("Not answerable")
