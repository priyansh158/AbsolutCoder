"""Console entry point for the AbsolutCoder terminal agent."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="absolut",
        description="AbsolutCoder plans coding work, edits files, runs approved commands, and commits.",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Run one task and exit. Omit to open the terminal loop.",
    )
    parser.add_argument(
        "--attach",
        action="append",
        default=[],
        metavar="PATH",
        help="Attach a file to a one-shot task. Images are sent as image blocks.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not _has_api_key():
        print(
            "OPENAI_API_KEY is not set. Set it to your OpenAI API key and run absolut again.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    from absolutcoder.repl import run_cli

    raise SystemExit(run_cli(args))


def _has_api_key() -> bool:
    import os

    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


if __name__ == "__main__":
    main()
