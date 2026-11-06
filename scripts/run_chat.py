"""Terminal chat for verifying the chat backend (Milestone 3)."""

import argparse

from app.core.exceptions import LocalMindError
from app.core.logging_config import setup_logging
from app.inference.generator import ChatSession
from app.inference.parameters import GenerationParams
from app.models.ollama_client import list_models


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--no-think", action="store_true", help="disable thinking (qwen3)")
    args = parser.parse_args()

    try:
        models = list_models()
    except LocalMindError as exc:
        print(f"[ERROR] {exc}")
        return
    print("Installed models:")
    for m in models:
        print(f"  {m.name:<28} {m.size_gb:>6.2f} GB  {m.parameter_size}  {m.quantization}")
    if not args.model:
        return

    params = GenerationParams(

        temperature=args.temperature,
        max_tokens=args.max_tokens,
        think=False if args.no_think else None,
    )
    session = ChatSession(model=args.model, params=params)
    print(f"\nChatting with {args.model}. Commands: /clear  /quit\n")

    while True:
        user = input("You: ").strip()
        if user == "/quit":
            break
        if user == "/clear":
            session.clear()
            print("[history cleared]")
            continue
        if not user:
            continue
        print("AI : ", end="", flush=True)
        try:
            for text in session.send_stream(user):
                print(text, end="", flush=True)
        except LocalMindError as exc:
            print(f"\n[ERROR] {exc}")
            continue
        s = session.last_stats
        if s:
            print(f"\n[{s.tokens_generated} tok | {s.tokens_per_second:.1f} tok/s | "
                  f"TTFT {s.ttft_s or 0:.2f}s | load {s.load_s:.2f}s]\n")


if __name__ == "__main__":
    main()