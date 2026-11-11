"""Minimal streaming inference against a local Ollama server."""

import argparse
import time

import ollama

DEFAULT_HOST = "http://localhost:11434"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prompt against a local model.")
    parser.add_argument("--model", required=True, help="e.g. llama3.2:3b")
    parser.add_argument(
        "--prompt", default="Explain what a Small Language Model is in two sentences."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    client = ollama.Client(host=DEFAULT_HOST)
    start = time.perf_counter()
    first_token_time = None
    final = None

    try:
        stream = client.chat(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            stream=True,
            options={"temperature": args.temperature, "num_predict": args.max_tokens},
        )
        for chunk in stream:
            text = chunk.message.content or ""

            if text and first_token_time is None:
                first_token_time = time.perf_counter() - start
            print(text, end="", flush=True)
            if chunk.done:
                final = chunk
    except ConnectionError:
        print("\n[ERROR] Cannot reach Ollama. Is the server running? Try: ollama list")
        return
    except ollama.ResponseError as exc:
        print(f"\n[ERROR] Ollama returned an error: {exc.error} (status {exc.status_code})")
        if exc.status_code == 404:
            print(f"Model not found. Download it with: ollama pull {args.model}")
        return

    total = time.perf_counter() - start
    print("\n\n--- Stats ---")
    if first_token_time is not None:
        print(f"Time to first token (client-measured): {first_token_time:.2f} s")
    print(f"Total wall time: {total:.2f} s")
    if final is not None and final.eval_count and final.eval_duration:
        gen_s = final.eval_duration / 1e9  # nanoseconds -> seconds
        print(f"Generated tokens: {final.eval_count}")
        print(f"Generation time : {gen_s:.2f} s")
        print(f"Tokens/second   : {final.eval_count / gen_s:.1f}")
        print(f"Model load time : {(final.load_duration or 0) / 1e9:.2f} s")
        print(f"Prompt eval time: {(final.prompt_eval_duration or 0) / 1e9:.2f} s")


if __name__ == "__main__":
    main()
