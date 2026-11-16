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
    parser.add_argument(
        "--until",
        metavar="COMMAND",
        help="After the task, run COMMAND and keep fixing until it passes (this run only).",
    )
    parser.add_argument(
        "--max-rounds",
        type=_positive_int,
        metavar="N",
        help="Stop the --until loop after N turns.",
    )
    return parser


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.task:
        if args.until is not None:
            parser.error("--until needs a task")
        if args.max_rounds is not None:
            parser.error("--max-rounds needs a task")
    if args.until is not None and not args.until.strip():
        parser.error("--until needs a command")
    import os

    from absolutcoder.config import ensure_config
    from absolutcoder.models import missing_keys

    try:
        config = ensure_config()
    except (ValueError, KeyError) as error:
        detail = f"missing key {error.args[0]}" if isinstance(error, KeyError) else error
        print(f"Config error: {detail}", file=sys.stderr)
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
