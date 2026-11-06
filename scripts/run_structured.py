"""CLI to test structured output (Milestone 4)."""

import argparse

from app.core.exceptions import LocalMindError
from app.core.logging_config import setup_logging
from app.structured_output.generator import generate_structured
from app.structured_output.schemas import SCHEMAS

SAMPLES = {
    "ticket": "I was charged twice for my subscription this month and nobody has replied "
              "to my emails. This is unacceptable, fix it today.",
    "resume": "Priya Nair, priya@example.com. Data analyst with 3 years at Infosys. "
              "B.Tech in Computer Science, VIT, 2021. Skills: Python, SQL, Power BI.",
    "meeting": "Weekly sync on 12 March. Present: Ravi, Meera, John. We decided to delay the "
               "launch to April. Meera will update the roadmap by Friday.",
}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--schema", choices=list(SCHEMAS), default="ticket")
    parser.add_argument("--text", help="input text (defaults to a built-in sample)")
    parser.add_argument("--unconstrained", action="store_true", help="skip schema-constrained decoding")
    args = parser.parse_args()

    text = args.text or SAMPLES[args.schema]
    try:
        result = generate_structured(
            args.model, args.schema, text, constrained=not args.unconstrained

        )
    except LocalMindError as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"Attempts: {result.attempts} | constrained: {result.constrained}")
    for i, err in enumerate(result.errors, 1):
        print(f"  attempt {i} failed: {err}")
    if result.ok:
        print(result.data.model_dump_json(indent=2))
    else:
        print(f"FAILED after {result.attempts} attempts. Last raw output:\n{result.raw_outputs[-1]}")


if __name__ == "__main__":
    main()