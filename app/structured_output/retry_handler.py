"""Builds the corrective conversation for a retry."""


def build_retry_messages(
    messages: list[dict[str, str]], bad_output: str, error: str
) -> list[dict[str, str]]:
    """Append the failed answer plus the error so the model can correct itself.

    Resending the identical prompt at temperature 0 would return the identical
    (invalid) output, so a retry must change the input. Feeding back the error does that.
    """
    return [
        *messages,
        {"role": "assistant", "content": bad_output},
        {
            "role": "user",
            "content": (
                f"Your previous output was invalid. Problems: {error}\n"
                "Return ONLY the corrected JSON object, with no extra text."
            ),
        },
    ]
