
from app.evaluation.quality_metrics import (
    evaluate,
    score_answer_any,
    score_contains_all,
    score_json_fields,
    score_label,
)
from app.structured_output.schemas import TicketClassification

LABELS = ["positive", "negative", "neutral"]


def test_label_exact_and_punctuation():
    assert score_label("Positive.", "positive", LABELS) == 1.0


def test_label_penalizes_hedging_between_labels():
    assert score_label("not positive, more negative", "positive", LABELS) == 0.0


def test_spam_label_not_confused_with_other_words():
    assert score_label("This is spam", "spam", ["spam", "legitimate"]) == 1.0


def test_answer_is_whole_word():
    assert score_answer_any("The answer is 18.", ["18"]) == 1.0
    assert score_answer_any("180", ["18"]) == 0.0


def test_contains_all_with_alternatives_and_length_cap():
    assert score_contains_all("Moving to Bangalore", ["bengaluru|bangalore"]) == 1.0
    assert score_contains_all("only one", ["one", "two"]) == 0.5
    assert score_contains_all("one two " + "word " * 50, ["one", "two"], max_words=10) == 0.5


def test_json_fields_partial_credit_and_any_of():
    data = TicketClassification(
        category="Billing", priority="High", sentiment="Neutral", summary="Charged twice."
    )
    score, detail = score_json_fields(data, {"category": "Billing", "sentiment": "Negative"})
    assert score == 0.5 and "sentiment" in detail
    ok, _ = score_json_fields(data, {"category": {"any_of": ["Account", "Billing"]}})
    assert ok == 1.0


def test_json_fields_without_data_scores_zero():
    assert score_json_fields(None, {"category": "Billing"})[0] == 0.0


def test_evaluate_dispatch():
    item = {"eval": {"type": "answer_any", "answers": ["paris"]}}
    assert evaluate(item, "Paris", None)[0] == 1.0
    assert evaluate({}, "x", None)[0] is None
