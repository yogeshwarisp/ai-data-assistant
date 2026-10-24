from evaluation.run_eval import check_gold, evaluate, load_questions, results_match
from src.config import Settings
from src.text_to_sql import answer_question  # noqa: F401  (ensures import path works)
import pandas as pd


def test_question_set_has_enough_questions():
    qs = load_questions()
    assert len(qs) >= 30
    assert any(not q["answerable"] for q in qs)


def test_all_gold_queries_run():
    assert check_gold() == []


def test_results_match_ignores_order_and_float_noise():
    a = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.004]})
    b = pd.DataFrame({"n": ["b", "a"], "v": [2.0, 1.0]})
    assert results_match(a, b, ordered=False)
    assert not results_match(a, b, ordered=True)


class GoldClient:
    """Perfect 'model': looks the gold SQL up from the question text."""

    def __init__(self):
        self.responses = self
        self.gold = {q["question"]: q["gold_sql"] for q in load_questions()}

    def create(self, input, **kwargs):
        question = input.rsplit("Question: ", 1)[1].split("\n")[0]
        sql = self.gold[question]
        text = f"```sql\n{sql}\n```" if sql else "CANNOT_ANSWER: not in data"
        return type("R", (), {"output_text": text})()


def test_harness_scores_a_perfect_model_as_perfect():
    import evaluation.run_eval as ev

    settings = Settings("k", "m", 20, 500000, 10000, 10)
    original = ev.get_settings
    ev.get_settings = lambda: settings
    try:
        report = evaluate(client=GoldClient())
    finally:
        ev.get_settings = original
    assert report["summary"]["execution_accuracy"] == 1.0
    assert report["summary"]["correct_refusal_rate"] == 1.0
