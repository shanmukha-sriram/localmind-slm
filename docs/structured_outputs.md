# Structured outputs

## Three levels of validity
1. **Syntactic:** is it parseable JSON? (`json.loads`)
2. **Schema:** right fields, types, allowed values? (Pydantic)
3. **Semantic:** is the content actually correct? **Neither of the above checks this.**
A ticket about a broken laptop labelled `Billing` passes levels 1 and 2.

## Mechanism
Pydantic model -> `model_json_schema()` -> Ollama `format` (constrained decoding restricts which
tokens may be sampled, so structure is almost guaranteed) -> parse -> validate. On failure, one
retry that appends the invalid output and the error text, because resending an identical prompt
at temperature 0 would reproduce the same failure.

## Constrained vs unconstrained
Unconstrained mode relies on the prompt alone and exposes real validation failures. It is used
to measure how much constrained decoding helps.

## Results
Tested with llama3.2:3b on the support ticket classification sample ("I was charged twice for my subscription this month and nobody has replied to my emails. This is unacceptable, fix it today."):

Attempts: 1 (passed on the first try)

Schema Validity: Yes (valid JSON matching the Pydantic schema)

Total Wall Time: 3.8s

Generated Output:

JSON
{
  "category": "Billing",
  "priority": "High",
  "sentiment": "Negative",
  "summary": "Duplicate charge and unresponsive support"
}

## Limits
Constrained decoding fixes structure, not truth. Truncation at `max_tokens` can still produce
invalid JSON. Models can fill a valid field with a wrong value.