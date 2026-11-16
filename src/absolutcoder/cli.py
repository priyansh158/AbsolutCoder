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
    import os

    from absolutcoder.config import ensure_config
    from absolutcoder.models import missing_keys

    try:
        config = ensure_config()
    except ValueError as error:
        print(f"Config error: {error}", file=sys.stderr)
        raise SystemExit(1)
    missing = missing_keys(config, os.environ)
    if missing:
        for entry in missing:
            print(f"{entry} is not set. Set it and run absolut again.", file=sys.stderr)
        raise SystemExit(1)
    from absolutcoder.repl import run_cli

    raise SystemExit(run_cli(args, config=config))


if __name__ == "__main__":
    main()
