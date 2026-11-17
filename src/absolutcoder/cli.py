"""Console entry point for the AbsolutCoder terminal agent."""

from __future__ import annotations

import argparse
import sys

_EPILOG = """\
In the terminal loop:
  /model <id>       switch the main model        /subagent <id>  switch the worker model
  /auto [on|off]    run commands without asking  /resume [N]     resume an earlier session
  /attach <path>    attach a file to the next message
  /agents           list the worker roles        /tasks          list background tasks
  /calls            show every call of the last turn (while it works: down arrow shows them)
  /skills           list, add or remove global skills
  /<skill-name> <request>     run a request with that skill's instructions loaded
  /map [build|update|on|off]  code map of this repository
  /verify <cmd|off> check command run after every request
  /loop <N>(s|m|h) <prompt>   repeat a prompt    /exit, /quit    leave
  /help             list these commands

Settings live in config.json in ~/.absolutcoder (or in $ABSOLUTCODER_HOME when set).
Model ids are provider:model, for example anthropic:claude-sonnet-4-5; a bare id means OpenAI.

--until COMMAND reruns the task until COMMAND passes; --auto runs commands without
asking for this run only (the git policy still applies).
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="absolut",
        description="AbsolutCoder plans coding work, edits files, runs approved commands, and commits.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument(
        "--until-timeout",
        type=_positive_int,
        metavar="N",
        help="Give the check command N seconds per run (this run only; default verify_timeout, 1800).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run commands without asking, for this run only (not saved). The git policy still applies.",
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


def utf8_when_piped(stream) -> None:
    """Write UTF-8 (replacing what cannot be encoded) when ``stream`` is not a terminal.

    A Windows pipe defaults to cp1252, where characters such as ✓ or → raise.
    """

    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        if stream.isatty():
            return
    except (AttributeError, ValueError, OSError):
        pass
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        utf8_when_piped(stream)
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

    from absolutcoder.config import config_path, ensure_config
    from absolutcoder.models import missing_keys

    try:
        config = ensure_config()
    except EOFError:
        print(
            f"No config found at {config_path()}. Run absolut once in a terminal to choose models, "
            'or create that file with "main_model" and "subagent_model".',
            file=sys.stderr,
        )
        raise SystemExit(1)
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
