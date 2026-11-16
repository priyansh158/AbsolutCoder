"""Repeat work: re-run a check after each turn, or repeat a prompt on an interval.

Both loops take the turn runner as a plain callable so they can be driven by
fakes. A turn runner returns ``None`` when the turn failed.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from deepagents.backends.protocol import ExecuteResponse

_INTERVAL = re.compile(r"([0-9]+)([smh])", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600}
_TAIL_CHARS = 4000


@dataclass(frozen=True)
class VerifyOutcome:
    passed: bool
    rounds: int  # turns that were run
    last_output: str  # output of the last check, or the reason the loop stopped


def parse_interval(text: str) -> float:
    """``30s``, ``5m`` or ``2h`` as seconds; anything else raises ValueError."""

    match = _INTERVAL.fullmatch(text.strip())
    if match is None or int(match.group(1)) == 0:
        raise ValueError(f"not an interval: {text!r} (use <N>s, <N>m or <N>h)")
    return float(max(1, int(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]))


def failure_prompt(command: str, exit_code: int | None, output: str) -> str:
    """The follow-up message that hands a failed check back to the model."""

    tail = output[-_TAIL_CHARS:]
    return (
        f"The check command failed (exit code {exit_code}). Find the cause, fix it, and make the "
        "check pass. Do not weaken or delete the check.\n\n"
        f"Command: {command}\n\n"
        f"Output (last {len(tail)} characters):\n"
        f"{tail}"
    )


def run_until_verified(
    run_turn: Callable[[str], object],
    verify: Callable[[], ExecuteResponse],
    prompt: str,
    *,
    command: str = "",
    max_rounds: int = 5,
) -> VerifyOutcome:
    """Run a turn, then the check; feed each failure back until it passes or rounds run out.

    ``KeyboardInterrupt`` and any exception from ``verify`` reach the caller.
    """

    limit = max(1, max_rounds)
    message = prompt
    rounds = 0
    while True:
        rounds += 1
        if run_turn(message) is None:
            return VerifyOutcome(False, rounds, "turn failed")
        result = verify()
        if result.exit_code == 0:
            return VerifyOutcome(True, rounds, result.output)
        if rounds >= limit:
            return VerifyOutcome(False, rounds, result.output)
        message = failure_prompt(command, result.exit_code, result.output)


def run_interval_loop(
    run_turn: Callable[[str], object],
    prompt: str,
    interval: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_iterations: int | None = None,
    on_iteration: Callable[[int], None] | None = None,
) -> int:
    """Run ``prompt`` every ``interval`` seconds until Ctrl+C or ``max_iterations``.

    Returns the number of iterations started.
    """

    started = 0
    try:
        while max_iterations is None or started < max_iterations:
            started += 1
            if on_iteration is not None:
                on_iteration(started)
            run_turn(prompt)
            if max_iterations is not None and started >= max_iterations:
                break
            sleep(interval)
    except KeyboardInterrupt:
        pass
    return started
