from app.structured_output.schemas import TicketClassification
from app.structured_output.validator import validate_output

VALID = (
    '{"category": "Billing", "priority": "High", '
    '"sentiment": "Negative", "summary": "Charged twice."}'
)


def test_valid_output():
    out = validate_output(VALID, TicketClassification)
    assert out.ok and out.data.category == "Billing"


def test_code_fences_are_stripped():
    assert validate_output(f"```json\n{VALID}\n```", TicketClassification).ok


def test_invalid_json_reports_json_stage():
    out = validate_output("not json at all", TicketClassification)
    assert not out.ok and out.stage == "json"


def test_valid_json_but_bad_enum_reports_schema_stage():
    bad = VALID.replace("Billing", "Refunds")
    out = validate_output(bad, TicketClassification)
    assert not out.ok and out.stage == "schema" and "category" in out.error


def test_missing_field_reports_schema_stage():
    out = validate_output('{"category": "Billing"}', TicketClassification)
    assert out.stage == "schema" and "priority" in out.error


def test_semantically_wrong_but_valid_output_passes():
    # Documents the limitation: schema validation cannot judge correctness.
    wrong = VALID.replace("Billing", "Shipping")
    assert validate_output(wrong, TicketClassification).ok
